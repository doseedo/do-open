#!/usr/bin/env python
"""
Analyze the temporal dependency graph of discovered compositions.

This reveals which musical operations are prerequisites for others,
showing the hierarchical structure of compositional complexity.

Usage:
    python scripts/analyze_discovery_dependency_graph.py discovery_results_cpu.npy
"""

import numpy as np
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def parse_composition_name(name: str) -> Tuple[str, str]:
    """Extract component transforms from composition name."""
    if '_o_' in name:
        parts = name.split('_o_')
        return parts[0], parts[1] if len(parts) > 1 else ''
    return name, ''


def build_dependency_graph(iteration_history: List[Dict]) -> Dict:
    """
    Build dependency graph showing which compositions enable others.

    A composition C discovered in iteration i depends on compositions
    A and B if C = A ∘ B and both A and B were discovered before iteration i.
    """
    # Track when each transform was discovered
    discovery_time = {}  # transform_name -> iteration
    transform_to_iteration = {}

    # Primitives discovered in iteration 0
    if iteration_history:
        first_iter = iteration_history[0]
        if 'discovered_compositions' in first_iter:
            # Get primitives from first compositions
            for comp in first_iter['discovered_compositions']:
                name = comp['name']
                component1, component2 = parse_composition_name(name)
                if component1 and component1 not in discovery_time:
                    discovery_time[component1] = 0
                if component2 and component2 not in discovery_time:
                    discovery_time[component2] = 0

    # Track discovered compositions by iteration
    dependencies = defaultdict(list)  # composition -> [dependencies]
    enabled_by = defaultdict(list)  # composition -> [enabled_compositions]

    for iter_data in iteration_history:
        iteration = iter_data['iteration']

        if 'discovered_compositions' not in iter_data:
            continue

        for comp in iter_data['discovered_compositions']:
            name = comp['name']
            discovery_time[name] = iteration
            transform_to_iteration[name] = iter_data

            # Check if this composition depends on previously discovered ones
            component1, component2 = parse_composition_name(name)

            deps = []
            if component1 in discovery_time and discovery_time[component1] < iteration:
                deps.append(component1)
            if component2 in discovery_time and discovery_time[component2] < iteration:
                deps.append(component2)

            if deps:
                dependencies[name] = deps
                for dep in deps:
                    enabled_by[dep].append(name)

    return {
        'discovery_time': discovery_time,
        'dependencies': dict(dependencies),
        'enabled_by': dict(enabled_by),
        'transform_to_iteration': transform_to_iteration
    }


def analyze_dependency_patterns(graph: Dict):
    """Analyze and print dependency patterns."""
    discovery_time = graph['discovery_time']
    dependencies = graph['dependencies']
    enabled_by = graph['enabled_by']

    print("\n" + "="*70)
    print("DEPENDENCY GRAPH ANALYSIS")
    print("="*70)

    # Group by iteration
    by_iteration = defaultdict(list)
    for name, iteration in discovery_time.items():
        by_iteration[iteration].append(name)

    print("\nDiscovery Timeline:")
    for iteration in sorted(by_iteration.keys()):
        compositions = by_iteration[iteration]
        print(f"\nIteration {iteration}: {len(compositions)} transforms")
        if iteration == 0:
            print("  (Primitives)")
        else:
            # Show top 5 by improvement
            comps_sorted = sorted(compositions, key=lambda x: len(enabled_by.get(x, [])), reverse=True)
            for comp in comps_sorted[:5]:
                deps = dependencies.get(comp, [])
                enables = enabled_by.get(comp, [])
                print(f"  • {comp}")
                if deps:
                    print(f"    Depends on: {', '.join(deps)}")
                if enables:
                    print(f"    Enables {len(enables)} future compositions")

    # Find "keystone" compositions - ones that enable many others
    print("\n" + "="*70)
    print("KEYSTONE COMPOSITIONS (enable most future discoveries)")
    print("="*70)

    keystones = sorted(enabled_by.items(), key=lambda x: len(x[1]), reverse=True)
    for comp, enabled in keystones[:10]:
        print(f"\n{comp} (iteration {discovery_time[comp]})")
        print(f"  Enables {len(enabled)} compositions:")
        for e in enabled[:5]:
            print(f"    → {e} (iteration {discovery_time[e]})")
        if len(enabled) > 5:
            print(f"    ... and {len(enabled)-5} more")

    # Find "late bloomers" - compositions that depend on many others
    print("\n" + "="*70)
    print("COMPLEX COMPOSITIONS (depend on most prior discoveries)")
    print("="*70)

    complex_comps = sorted(dependencies.items(), key=lambda x: len(x[1]), reverse=True)
    for comp, deps in complex_comps[:10]:
        print(f"\n{comp} (iteration {discovery_time[comp]})")
        print(f"  Depends on {len(deps)} compositions:")
        for d in deps[:5]:
            print(f"    ← {d} (iteration {discovery_time[d]})")
        if len(deps) > 5:
            print(f"    ... and {len(deps)-5} more")

    # Calculate "depth" - longest dependency chain
    print("\n" + "="*70)
    print("DEPENDENCY DEPTH")
    print("="*70)

    def get_depth(comp, memo={}):
        if comp in memo:
            return memo[comp]
        deps = dependencies.get(comp, [])
        if not deps:
            memo[comp] = 0
            return 0
        depth = 1 + max(get_depth(d, memo) for d in deps)
        memo[comp] = depth
        return depth

    depths = {comp: get_depth(comp) for comp in discovery_time.keys()}
    by_depth = defaultdict(list)
    for comp, depth in depths.items():
        by_depth[depth].append(comp)

    for depth in sorted(by_depth.keys(), reverse=True)[:5]:
        comps = by_depth[depth]
        print(f"\nDepth {depth}: {len(comps)} compositions")
        for comp in comps[:3]:
            print(f"  • {comp} (iteration {discovery_time[comp]})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_discovery_dependency_graph.py <results.npy>")
        sys.exit(1)

    results_file = sys.argv[1]
    print(f"Loading results from {results_file}...")

    results = np.load(results_file, allow_pickle=True).item()

    if 'iteration_history' not in results:
        print("Error: No iteration_history found in results")
        sys.exit(1)

    iteration_history = results['iteration_history']

    print(f"Loaded {len(iteration_history)} iterations")
    print(f"Total transforms: {results.get('all_transforms', 'N/A')}")

    # Build and analyze dependency graph
    graph = build_dependency_graph(iteration_history)
    analyze_dependency_patterns(graph)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()
