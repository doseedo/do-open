import { SignUp } from '@clerk/nextjs';

export default function Page() {
  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--gradient-glow-subtle, #0a0a0f)',
        padding: '2rem',
      }}
    >
      <SignUp
        path="/sign-up"
        routing="path"
        signInUrl="/sign-in"
        forceRedirectUrl="/studio"
        fallbackRedirectUrl="/studio"
      />
    </div>
  );
}
