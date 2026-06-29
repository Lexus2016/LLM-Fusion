import { describe, it, expect } from "vitest";
import { stripHopByHopHeaders } from "../src/headers";

describe("stripHopByHopHeaders", () => {
  it("deletes content-length, content-encoding, transfer-encoding and keeps the rest", () => {
    const headers = new Headers({
      "content-type": "application/json",
      "content-length": "999",
      "content-encoding": "gzip",
      "transfer-encoding": "chunked",
      "x-fusion-usage": '{"calls":1,"total":15}',
      "cache-control": "no-cache",
    });
    stripHopByHopHeaders(headers);
    expect(headers.get("content-length")).toBeNull();
    expect(headers.get("content-encoding")).toBeNull();
    expect(headers.get("transfer-encoding")).toBeNull();
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("x-fusion-usage")).toBe('{"calls":1,"total":15}');
    expect(headers.get("cache-control")).toBe("no-cache");
  });

  it("is case-insensitive (HTTP headers)", () => {
    const headers = new Headers({ "Content-Length": "10", "CONTENT-ENCODING": "gzip" });
    stripHopByHopHeaders(headers);
    expect(headers.get("content-length")).toBeNull();
    expect(headers.get("content-encoding")).toBeNull();
  });
});