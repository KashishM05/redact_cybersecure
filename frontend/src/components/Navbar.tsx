// import { ArrowRight, CircleArrowLeft, CircleArrowRight, Shield } from "lucide-react";
// import Link from "next/link";
// import React from "react";

// const Navbar = () => {
//   return (
//     <header>
//       <nav className="relative z-10 flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
//         <div className="flex items-center space-x-2">
//           <span className="text-2xl font-bold">Cyber secure</span>
//         </div>
//         <div className="hidden md:flex items-center space-x-8 text-sm">
//           <Link href="#" className="hover:text-blue-400 transition">
//             Home
//           </Link>
//           <Link href="/dashboard" className="hover:text-blue-400 transition">
//             Dashboard
//           </Link>
//           <Link href="#" className="hover:text-blue-400 transition">
//             Live monitoring
//           </Link>
//           <Link href="#" className="text-green-400">
//             Report
//           </Link>
//           <Link href="#how-it-works" className="hover:text-blue-400 transition">
//             How it works
//           </Link>
//         </div>
//         <button className="cursor-pointer font-bold transition-all duration-200 px-5 py-2.5 rounded-full bg-lime-300 border border-transparent flex items-center text-sm hover:bg-lime-400 active:scale-95 text-black group">
//           <span>Continue</span>
//           <CircleArrowRight className="w-6 h-6 ml-2 transition-transform duration-300 group-hover:translate-x-1" />
//         </button>
//       </nav>
//     </header>
//   );
// };

// export default Navbar;

import { CircleArrowRight } from "lucide-react";
import Link from "next/link";
import React from "react";

const Navbar = () => {
  return (
    <header className="fixed top-0 left-0 w-full z-50  backdrop-blur-lg border-b border-white/10">
      <nav className="flex items-center justify-between px-8 py-4 max-w-7xl mx-auto">
        
        {/* Logo */}
        <div className="flex items-center space-x-2">
          <span className="text-2xl font-bold text-white">Cyber secure</span>
        </div>

        {/* Desktop Menu */}
        <div className="hidden md:flex items-center space-x-8 text-sm">
          <Link href="#" className="hover:text-blue-400 transition text-white">
            Home
          </Link>
          <Link href="/dashboard" className="hover:text-blue-400 transition text-white">
            Dashboard
          </Link>
          <Link href="#" className="hover:text-blue-400 transition text-white">
            Live monitoring
          </Link>
          <Link href="#" className="text-green-400">
            Report
          </Link>
          <Link
            href="#how-it-works"
            className="hover:text-blue-400 transition text-white"
          >
            How it works
          </Link>
        </div>

        {/* CTA Button */}
        <button className="cursor-pointer font-bold transition-all duration-200 px-5 py-2.5 rounded-full bg-lime-300 border border-transparent flex items-center text-sm hover:bg-lime-400 active:scale-95 text-black group">
          <span>Continue</span>
          <CircleArrowRight className="w-6 h-6 ml-2 transition-transform duration-300 group-hover:translate-x-1" />
        </button>
      </nav>
    </header>
  );
};

export default Navbar;
