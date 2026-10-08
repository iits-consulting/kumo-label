import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import svelte from "eslint-plugin-svelte";
import svelteConfig from "./svelte.config.js";

export default tseslint.config(
  { ignores: ["dist"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...svelte.configs.recommended,
  {
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
  },
  {
    files: ["**/*.svelte", "**/*.svelte.ts", "**/*.svelte.js"],
    languageOptions: {
      parserOptions: {
        parser: tseslint.parser,
        extraFileExtensions: [".svelte"],
        svelteConfig,
      },
    },
  },
  {
    rules: {
      "@typescript-eslint/no-unused-vars": "off",
      // Plain Map/Set are deliberately non-reactive in this codebase (session
      // override maps, guards, frozen queues) — a correctness requirement.
      "svelte/prefer-svelte-reactivity": "off",
      // svelte-ignore comments target svelte-check's compiler warnings, which
      // eslint-plugin-svelte does not reproduce 1:1.
      "svelte/no-unused-svelte-ignore": "off",
      // Reset-on-taskConfig-change effects are deliberate (mirror React deps).
      "svelte/prefer-writable-derived": "off",
    },
  },
);
