import { redirect } from 'next/navigation';

// Legacy /login URL — redirect to the Clerk sign-in route.
export default function LoginRedirect() {
  redirect('/sign-in');
}
