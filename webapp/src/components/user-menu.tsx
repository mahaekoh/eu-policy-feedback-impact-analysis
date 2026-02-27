"use client";

import { useSession, signIn, signOut } from "next-auth/react";
import { Button } from "@/components/ui/button";

export function UserMenu() {
  const { data: session, status } = useSession();

  if (status === "loading") {
    return <div className="h-9 w-20 animate-pulse rounded-md bg-muted" />;
  }

  if (!session?.user) {
    return (
      <Button variant="ghost" size="sm" onClick={() => signIn("google")}>
        Sign in
      </Button>
    );
  }

  return (
    <div className="flex items-center gap-3">
      {session.user.image && (
        <img
          src={session.user.image}
          alt=""
          className="size-8 rounded-full"
          referrerPolicy="no-referrer"
        />
      )}
      <span className="text-sm font-medium hidden sm:inline">
        {session.user.name}
      </span>
      <Button variant="ghost" size="sm" onClick={() => signOut()}>
        Sign out
      </Button>
    </div>
  );
}
