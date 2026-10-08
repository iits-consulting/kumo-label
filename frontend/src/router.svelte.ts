// Minimal hand-rolled SPA router (spec §7): a $state wrapper around
// location.pathname + pushState/popstate. Route resolution lives in App.svelte.

const route = $state({ pathname: window.location.pathname });

window.addEventListener("popstate", () => {
  route.pathname = window.location.pathname;
});

export function navigate(to: string, opts: { replace?: boolean } = {}) {
  if (opts.replace) {
    window.history.replaceState({}, "", to);
  } else {
    window.history.pushState({}, "", to);
  }
  route.pathname = window.location.pathname;
}

export { route };
