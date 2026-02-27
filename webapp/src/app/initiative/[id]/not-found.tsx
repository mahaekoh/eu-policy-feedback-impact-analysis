import Link from "next/link";

export default function NotFound() {
  return (
    <main className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">Initiative not found</h1>
        <p className="text-muted-foreground mb-6">
          The initiative you are looking for does not exist.
        </p>
        <Link
          href="/"
          className="text-primary hover:underline"
        >
          &larr; Back to initiatives
        </Link>
      </div>
    </main>
  );
}
