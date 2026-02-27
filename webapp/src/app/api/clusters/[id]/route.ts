import { NextRequest, NextResponse } from "next/server";
import { getClusterData } from "@/lib/data";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const scheme = request.nextUrl.searchParams.get("scheme");

  if (!scheme) {
    return NextResponse.json({ error: "Missing scheme parameter" }, { status: 400 });
  }

  const data = await getClusterData(id, scheme);
  if (!data) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  return NextResponse.json(data);
}
