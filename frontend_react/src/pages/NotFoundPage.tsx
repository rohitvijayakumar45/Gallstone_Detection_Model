import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <div className="max-w-md mx-auto px-6 py-24 text-center">
      <p className="label-caps mb-2">404</p>
      <h2 className="font-display text-3xl mb-4">Route not found</h2>
      <Link to="/" className="text-accent hover:underline underline-offset-4">
        Back to overview
      </Link>
    </div>
  );
}
