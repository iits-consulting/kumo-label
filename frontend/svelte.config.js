import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

export default {
  preprocess: [
    // ponytail: layerchart 2.x ships native `@layer base/components` CSS, which the
    // Tailwind v3 PostCSS plugin rejects ("no matching @tailwind directive"). Renaming
    // the layer keeps identical cascade behavior; drop this when moving to Tailwind v4.
    {
      name: "layerchart-tailwind3-layer-rename",
      style({ content, filename }) {
        if (filename?.includes("node_modules/layerchart")) {
          return {
            code: content.replace(
              /@layer\s+(base|components|utilities)\b/g,
              "@layer layerchart",
            ),
          };
        }
      },
    },
    vitePreprocess(),
  ],
};
