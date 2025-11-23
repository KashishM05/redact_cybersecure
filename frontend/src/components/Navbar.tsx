"use client";

import { CircleArrowRight } from "lucide-react";
import Link from "next/link";
import React from "react";
import { usePathname } from "next/navigation";

const Navbar = () => {
  const pathname = usePathname();

  // Helper to check active link
  const isActive = (path: string) => pathname === path;

  return (
    <header className="fixed top-0 left-0 w-full z-50 backdrop-blur-lg border-white/10">
      <nav className="flex items-center justify-between px-8 py-2 max-w-7xl mx-auto">

        {/* Logo */}
        <div className="flex items-center space-x-2">
          <span className="text-2xl font-bold text-white">Cyber secure</span>
        </div>

        {/* Desktop Menu */}
        <div className="hidden md:flex items-center space-x-8 text-sm">

          <Link
            href="/"
            className={`transition ${
              isActive("/")
                ? "text-blue-400 border-b-2 border-blue-400 pb-1"
                : "text-white hover:text-blue-400"
            }`}
          >
            Home
          </Link>

          <Link
            href="/dashboard"
            className={`transition ${
              isActive("/dashboard")
                ? "text-blue-400 border-b-2 border-blue-400 pb-1"
                : "text-white hover:text-blue-400"
            }`}
          >
            Dashboard
          </Link>

          <Link
            href="/analyze"
            className={`transition ${
              isActive("/analyze")
                ? "text-blue-400 border-b-2 border-blue-400 pb-1"
                : ""
            }`}
          >
            Analyze
          </Link>

          {/* <Link
            href="/blockchain"
            className={`transition ${
              isActive("/blockchain")
                ? "text-blue-400 border-b-2 border-blue-400 pb-1"
                : "text-white hover:text-blue-400"
            }`}
          >
            Blockchain
          </Link> */}

          <Link
            href="/blockchain-demo"
            className={`transition ${
              isActive("/blockchain-demo")
                ? "text-blue-400 border-b-2 border-blue-400 pb-1"
                : "text-white hover:text-blue-400"
            }`}
          >
            Blockchain
          </Link>

          <Link
            href="/guide"
            className={`transition ${
              isActive("/guide")
                ? "text-blue-400 border-b-2 border-blue-400 pb-1"
                : "text-white hover:text-blue-400"
            }`}
          >
            Guide
          </Link>

        </div>
      </nav>
    </header>
  );
};

export default Navbar;
