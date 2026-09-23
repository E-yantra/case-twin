import { Link, useLocation } from "react-router-dom";
import { Plus, LayoutDashboard, Clock, FolderOpen, MessageSquare, Settings2 } from "lucide-react";

interface AppShellProps {
  breadcrumb: string;
  children: React.ReactNode;
}

export function AppShell({ breadcrumb, children }: AppShellProps) {
  const location = useLocation();
  const path = location.pathname;

  return (
    <div className="min-h-screen bg-zinc-50/30">
      <header className="sticky top-0 z-30 border-b border-zinc-200/80 bg-white/80 backdrop-blur-xl supports-[backdrop-filter]:bg-white/60">
        <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between px-4 sm:px-6">
          <div className="flex items-center gap-8">
            <Link to="/" className="group flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-[0.4rem] bg-gradient-to-tr from-zinc-900 to-zinc-800 text-white shadow-[0_1px_3px_rgba(0,0,0,0.1)] ring-1 ring-zinc-900/10 transition-transform duration-300 group-hover:scale-[1.03]">
                <span className="text-[13px] font-bold tracking-wider">CT</span>
              </div>
              <span className="text-[16px] font-semibold tracking-tight text-zinc-900">Case-Twin</span>
            </Link>

            <nav className="hidden md:flex items-center space-x-1">
              <Link to="/" className={`px-3 py-1.5 rounded-full text-[13px] font-medium transition-all duration-200 ${path === '/' ? 'bg-zinc-100/80 text-zinc-900 shadow-sm' : 'text-zinc-500 hover:text-zinc-900 hover:bg-zinc-50'}`}>
                <div className="flex items-center gap-1.5"><LayoutDashboard className="w-3.5 h-3.5" strokeWidth={2.5} /> Dashboard</div>
              </Link>
              <Link to="/history" className={`px-3 py-1.5 rounded-full text-[13px] font-medium transition-all duration-200 ${path === '/history' ? 'bg-zinc-100/80 text-zinc-900 shadow-sm' : 'text-zinc-500 hover:text-zinc-900 hover:bg-zinc-50'}`}>
                <div className="flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" strokeWidth={2.5} /> History</div>
              </Link>
              {/* <Link to="/cases" className={`px-3 py-1.5 rounded-full text-[13px] font-medium transition-all duration-200 ${path === '/cases' ? 'bg-zinc-100/80 text-zinc-900 shadow-sm' : 'text-zinc-500 hover:text-zinc-900 hover:bg-zinc-50'}`}>
                <div className="flex items-center gap-1.5"><FolderOpen className="w-3.5 h-3.5" strokeWidth={2.5} /> My Cases</div>
              </Link> */}
              <Link to="/chat" className={`px-3 py-1.5 rounded-full text-[13px] font-medium transition-all duration-200 ${path === '/chat' ? 'bg-zinc-100/80 text-zinc-900 shadow-sm' : 'text-zinc-500 hover:text-zinc-900 hover:bg-zinc-50'}`}>
                <div className="flex items-center gap-1.5"><MessageSquare className="w-3.5 h-3.5" strokeWidth={2.5} /> Chat</div>
              </Link>
            </nav>
          </div>

          <div className="flex items-center gap-4">
            {/* <div className="hidden sm:flex items-center gap-1.5">
              <a href="#" className="flex h-8 w-8 items-center justify-center rounded-full text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 transition-colors">
                <Settings2 className="h-4 w-4" />
              </a>
              <div className="w-px h-4 bg-zinc-200 mx-1" />
            </div> */}

            {breadcrumb && (
              <div className="hidden lg:flex items-center gap-2 text-[13px] font-medium">
                <span className="text-zinc-300">/</span>
                <span className="px-2 py-0.5 rounded-md text-zinc-500">{breadcrumb}</span>
              </div>
            )}

            {/* <Link to="/">
              <button className="flex items-center gap-1.5 rounded-full bg-zinc-900 px-4 py-1.5 text-[13px] font-medium text-white shadow-md shadow-zinc-900/10 hover:bg-zinc-800 transition-all active:scale-[0.98]">
                <Plus className="h-4 w-4" strokeWidth={2.5} />
                New Case
              </button>
            </Link> */}
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl px-4 py-7 sm:px-6 sm:py-9">{children}</main>
    </div>
  );
}
