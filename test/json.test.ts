import { describe, it, expect } from "vitest";
import { extractJsonObject, isJsonObjectString } from "../src/json";

describe("extractJsonObject", () => {
  it("returns a clean object unchanged", () => {
    expect(extractJsonObject('{"a":1}')).toBe('{"a":1}');
  });

  it("strips ```json code fences", () => {
    expect(extractJsonObject('```json\n{"route":"fusion"}\n```')).toBe('{"route":"fusion"}');
  });

  it("ignores leading prose and trailing commentary", () => {
    expect(extractJsonObject("Here is the analysis:\n{\"x\":1}\nHope that helps!")).toBe('{"x":1}');
  });

  it("handles nested objects", () => {
    expect(extractJsonObject('{"a":{"b":2},"c":3}')).toBe('{"a":{"b":2},"c":3}');
  });

  it("does not stop on braces inside string values", () => {
    expect(extractJsonObject('{"a":"} not the end {"}')).toBe('{"a":"} not the end {"}');
  });

  it("respects escaped quotes inside strings", () => {
    expect(extractJsonObject('{"a":"x\\"}y"}')).toBe('{"a":"x\\"}y"}');
  });

  it("returns null when there is no object", () => {
    expect(extractJsonObject("no json here")).toBeNull();
  });

  it("returns null for a truncated / unbalanced object", () => {
    expect(extractJsonObject('{"a":1')).toBeNull();
  });
});

describe("isJsonObjectString", () => {
  it("accepts only JSON objects — the one shape runnable tool arguments may take", () => {
    expect(isJsonObjectString("{}")).toBe(true);
    expect(isJsonObjectString('{"path":"a.txt"}')).toBe(true);
    // Scalars and arrays parse, but are not tool input.
    for (const s of ["5", "null", "true", '"a"', "[1,2]", "[]"]) {
      expect(isJsonObjectString(s)).toBe(false);
    }
    // Truncated / empty never parse.
    expect(isJsonObjectString('{"path":')).toBe(false);
    expect(isJsonObjectString("")).toBe(false);
  });
});
