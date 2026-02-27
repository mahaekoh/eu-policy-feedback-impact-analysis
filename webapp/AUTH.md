# Authentication Setup

Google sign-in via [Auth.js v5](https://authjs.dev/) (`next-auth@beta`). JWT-based sessions — no database required. The app is fully accessible without signing in.

## Architecture

```
src/auth.ts                          ← Central config (Google provider)
src/proxy.ts                         ← Session cookie refresh (Next.js 16 middleware proxy)
src/app/api/auth/[...nextauth]/route.ts  ← OAuth API routes
src/components/header.tsx            ← Nav bar with UserMenu
src/components/user-menu.tsx         ← Sign-in button / avatar + sign-out (client component)
src/app/layout.tsx                   ← SessionProvider wraps all pages
```

## Environment Variables

Set in `.env.local` (git-ignored):

| Variable | Description |
|---|---|
| `AUTH_SECRET` | Random secret for signing JWTs. Generate with `npx auth secret`. |
| `AUTH_GOOGLE_ID` | OAuth 2.0 Client ID from Google Cloud Console. |
| `AUTH_GOOGLE_SECRET` | OAuth 2.0 Client Secret from Google Cloud Console. |

## Google Cloud Console Setup

1. Go to [APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials)
2. Create an **OAuth 2.0 Client ID** (application type: Web application)
3. Add authorized redirect URI:
   - Local: `http://localhost:3000/api/auth/callback/google`
   - Production: `https://<your-domain>/api/auth/callback/google`
4. Copy the Client ID and Client Secret into `.env.local`

## Quick Start

```bash
cd webapp
npx auth secret          # writes AUTH_SECRET to .env.local
# Then add AUTH_GOOGLE_ID and AUTH_GOOGLE_SECRET to .env.local
npm run dev
```

## Session Access

**Client components** — use the `useSession` hook:

```tsx
"use client";
import { useSession } from "next-auth/react";

const { data: session } = useSession();
session?.user?.name   // Google display name
session?.user?.email  // Google email
session?.user?.image  // Google profile picture URL
```

**Server components / API routes** — use the `auth()` function:

```tsx
import { auth } from "@/auth";

const session = await auth();
session?.user?.email
```
