<script lang="ts">
  import { QueryClient, QueryClientProvider } from "@tanstack/svelte-query";
  import { Toaster } from "@/components/ui/sonner";
  import { TooltipProvider } from "@/components/ui/tooltip";
  import { route } from "@/router.svelte";
  import ProjectSetup from "@/pages/ProjectSetup.svelte";
  import Explorer from "@/pages/Explorer.svelte";
  import NotFound from "@/pages/NotFound.svelte";

  const queryClient = new QueryClient();

  const Page = $derived(
    route.pathname === "/"
      ? ProjectSetup
      : route.pathname.startsWith("/explorer")
        ? Explorer
        : NotFound,
  );
</script>

<QueryClientProvider client={queryClient}>
  <TooltipProvider>
    <Page />
    <Toaster />
  </TooltipProvider>
</QueryClientProvider>
