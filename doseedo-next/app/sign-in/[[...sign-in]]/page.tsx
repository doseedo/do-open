import { SignIn } from '@clerk/nextjs';

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
      <SignIn
        path="/sign-in"
        routing="path"
        signUpUrl="/sign-up"
        forceRedirectUrl="/studio"
        fallbackRedirectUrl="/studio"
      />
    </div>
  );
}
