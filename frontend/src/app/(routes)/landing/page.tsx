"use client";
// import React from 'react';
// import { Shield, Lock, Cloud, Zap, CheckCircle, Users } from 'lucide-react';

// export default function SkyFortLanding() {
//   return (
//     <div className="min-h-screen bg-[#0a0e27] text-white overflow-hidden">
//       {/* Animated gradient background */}
//       <div className="fixed inset-0 z-0">
//         <div className="absolute inset-0 bg-gradient-to-br from-[#0a0e27] via-[#1a1534] to-[#2d1b3d]"></div>
//         <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl animate-pulse"></div>
//         <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
//         <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '2s'}}></div>
//       </div>

//       {/* Navigation */}

//       {/* Hero Section */}
//       <section className="relative z-10 max-w-7xl mx-auto px-8 py-20 grid md:grid-cols-2 gap-12 items-center">
//         <div>
//           <p className="text-blue-400 text-sm mb-4 tracking-wider">SIMPLIFY YOUR SECURITY</p>
//           <h1 className="text-6xl font-bold mb-6 leading-tight">
//             SkyFort
//           </h1>
//           <p className="text-4xl font-light mb-4 text-gray-300">
//             The first cloud-firewall, built for WPFL
//           </p>
//           <p className="text-gray-400 mb-8 leading-relaxed">
//             SkyFort is simple and easy to use, no IT department required.
//             Your team is automatically secured by Fortress as
//             they work from anywhere in the world.
//           </p>
//           <div className="flex space-x-4">
//             <button className="bg-blue-600 hover:bg-blue-700 px-8 py-3 rounded-full font-semibold transition">
//               SIGN UP NOW
//             </button>
//             <button className="border border-gray-600 hover:border-blue-400 px-8 py-3 rounded-full font-semibold transition">
//               READ MORE
//             </button>
//           </div>
//         </div>

//         {/* 3D Isometric Illustration */}
//         <div className="relative h-96 flex items-center justify-center">
//           <div className="relative w-full h-full" style={{perspective: '1000px'}}>
//             {/* Cloud platform */}
//             <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-32 bg-gradient-to-br from-blue-600 to-purple-600 rounded-3xl shadow-2xl"
//                  style={{transform: 'translate(-50%, -50%) rotateX(60deg) rotateZ(-45deg)', boxShadow: '0 20px 60px rgba(59, 130, 246, 0.5)'}}>
//               <div className="absolute inset-0 flex items-center justify-center">
//                 <Cloud className="w-16 h-16 text-white/90" />
//               </div>
//               {/* Glowing orbs */}
//               <div className="absolute top-1/2 left-1/4 w-4 h-4 bg-blue-300 rounded-full animate-pulse"></div>
//               <div className="absolute top-1/3 right-1/4 w-3 h-3 bg-purple-300 rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
//             </div>

//             {/* Floating elements */}
//             <div className="absolute top-16 left-12 w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-lg animate-pulse"
//                  style={{transform: 'rotateX(45deg) rotateZ(-25deg)'}}>
//               <div className="flex items-center justify-center h-full">
//                 <Lock className="w-8 h-8 text-white" />
//               </div>
//             </div>

//             <div className="absolute bottom-16 right-12 w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl shadow-lg animate-pulse"
//                  style={{animationDelay: '1s', transform: 'rotateX(45deg) rotateZ(25deg)'}}>
//               <div className="flex items-center justify-center h-full">
//                 <Shield className="w-10 h-10 text-white" />
//               </div>
//             </div>

//             <div className="absolute top-1/4 right-16 w-12 h-12 bg-blue-400 rounded-full animate-pulse shadow-lg"
//                  style={{animationDelay: '1.5s'}}></div>
//           </div>
//         </div>
//       </section>

//       {/* Customer Logos */}
//       <section className="relative z-10 max-w-7xl mx-auto px-8 py-16">
//         <p className="text-center text-gray-400 mb-12">Trusted by 350+ happy customers</p>
//         <div className="grid grid-cols-3 md:grid-cols-7 gap-8 items-center opacity-40">
//           {['CGV', 'Hmall', 'pghm', 'LOTTE', 'NAVER', 'coupang', 'SK telecom', 'saramth', 'coupang', 'MBC', 'CGV', 'KBS'].map((company, i) => (
//             <div key={i} className="text-center text-gray-500 font-bold text-xl">
//               {company}
//             </div>
//           ))}
//         </div>
//       </section>

//       {/* Why SkyFort Section */}
//       <section className="relative z-10 max-w-7xl mx-auto px-8 py-20">
//         <div className="text-center mb-16">
//           <p className="text-blue-400 text-sm mb-4 tracking-wider">POWERFULL FEATURES</p>
//           <h2 className="text-4xl font-bold mb-4">Why SkyFort?</h2>
//           <p className="text-gray-400 max-w-2xl mx-auto">
//             SkyFort is the first and only easy to use cloud firewall
//             for remote teams.
//           </p>
//         </div>

//         <div className="grid md:grid-cols-3 gap-8">
//           <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/20 backdrop-blur-sm border border-blue-800/30 rounded-2xl p-8 hover:border-blue-600/50 transition">
//             <div className="bg-gradient-to-br from-orange-500 to-red-500 w-12 h-12 rounded-xl flex items-center justify-center mb-6">
//               <Shield className="w-6 h-6 text-white" />
//             </div>
//             <h3 className="text-xl font-bold mb-4">Reliable protection</h3>
//             <p className="text-gray-400 leading-relaxed">
//               We manage your network traffic, keeping
//               your site protected smoothly.
//             </p>
//           </div>

//           <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/20 backdrop-blur-sm border border-blue-800/30 rounded-2xl p-8 hover:border-blue-600/50 transition">
//             <div className="bg-gradient-to-br from-blue-500 to-cyan-500 w-12 h-12 rounded-xl flex items-center justify-center mb-6">
//               <Zap className="w-6 h-6 text-white" />
//             </div>
//             <h3 className="text-xl font-bold mb-4">Easy to set up</h3>
//             <p className="text-gray-400 leading-relaxed">
//               Your team's data is secured and
//               protected instantly.
//             </p>
//           </div>

//           <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/20 backdrop-blur-sm border border-blue-800/30 rounded-2xl p-8 hover:border-blue-600/50 transition">
//             <div className="bg-gradient-to-br from-orange-500 to-yellow-500 w-12 h-12 rounded-xl flex items-center justify-center mb-6">
//               <CheckCircle className="w-6 h-6 text-white" />
//             </div>
//             <h3 className="text-xl font-bold mb-4">Virus protection</h3>
//             <p className="text-gray-400 leading-relaxed">
//               All web traffic is intelligently screened for
//               malware and spyware.
//             </p>
//           </div>
//         </div>
//       </section>

//       {/* Features Grid */}
//       <section className="relative z-10 max-w-7xl mx-auto px-8 py-12">
//         <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
//           {[
//             { icon: Lock, title: 'Secure access', desc: 'Control who has access to your network' },
//             { icon: Shield, title: 'Network level protection', desc: 'Protect against DDoS attacks' },
//             { icon: Cloud, title: 'Cloud-native', desc: 'Built for modern cloud infrastructure' },
//             { icon: Users, title: 'Team collaboration', desc: 'Easy team management and access' },
//             { icon: Zap, title: 'Lightning fast', desc: 'Minimal latency, maximum speed' },
//             { icon: CheckCircle, title: 'Always available', desc: '99.9% uptime guarantee' },
//             { icon: Lock, title: 'Encrypted', desc: 'End-to-end encryption' },
//             { icon: Shield, title: '24/7 monitoring', desc: 'Round the clock security' }
//           ].map((feature, i) => (
//             <div key={i} className="bg-gradient-to-br from-blue-900/20 to-purple-900/10 backdrop-blur-sm border border-blue-800/20 rounded-xl p-6 hover:border-blue-600/40 transition">
//               <feature.icon className="w-8 h-8 text-blue-400 mb-4" />
//               <h4 className="font-semibold mb-2">{feature.title}</h4>
//               <p className="text-sm text-gray-400">{feature.desc}</p>
//             </div>
//           ))}
//         </div>
//       </section>

//       {/* CTA Section */}
//       <section className="relative z-10 max-w-4xl mx-auto px-8 py-20 text-center">
//         <h2 className="text-4xl font-bold mb-6">Ready to get started?</h2>
//         <p className="text-gray-400 mb-8 text-lg">
//           Join 350+ companies protecting their teams with SkyFort
//         </p>
//         <button className="bg-blue-600 hover:bg-blue-700 px-10 py-4 rounded-full text-lg font-semibold transition shadow-lg hover:shadow-blue-500/50">
//           Get Started Now
//         </button>
//       </section>

//       {/* Footer */}
//       <footer className="relative z-10 border-t border-gray-800 mt-20">
//         <div className="max-w-7xl mx-auto px-8 py-12">
//           <div className="flex flex-col md:flex-row justify-between items-center">
//             <div className="flex items-center space-x-2 mb-4 md:mb-0">
//               <Shield className="w-6 h-6 text-blue-400" />
//               <span className="text-xl font-bold">SkyFort</span>
//             </div>
//             <div className="text-gray-400 text-sm">
//               © 2024 SkyFort. All rights reserved.
//             </div>
//           </div>
//         </div>
//       </footer>
//     </div>
//   );
// }












// import React, { useEffect, useRef } from "react";
// import { Shield, Lock, Cloud, Zap, CheckCircle, Users } from "lucide-react";

// export default function SkyFortLanding() {
//   const vantaRef = useRef<HTMLDivElement | null>(null);
//   const vantaEffectRef = useRef<VantaEffect | null>(null);

//   useEffect(() => {
//     // 1. Load THREE from CDN
//     const threeScript = document.createElement("script");
//     threeScript.src =
//       "https://cdnjs.cloudflare.com/ajax/libs/three.js/r121/three.min.js";
//     threeScript.onload = () => {
//       // 2. Load VANTA from CDN
//       const vantaScript = document.createElement("script");
//       vantaScript.src =
//         "https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.globe.min.js";
//       vantaScript.onload = () => {
//         if (
//           !vantaEffectRef.current &&
//           (window as any).VANTA &&
//           vantaRef.current
//         ) {
//           vantaEffectRef.current = (window as any).VANTA.GLOBE({
//             el: vantaRef.current,
//             mouseControls: true,
//             touchControls: true,
//             gyroControls: false,
//             minHeight: 200,
//             minWidth: 200,
//             scale: 1.0,
//             scaleMobile: 1.0,
//             color: 0x543b54,
//             backgroundColor: 0x1f1f1f,
//           }) as VantaEffect;
//         }
//       };

//       document.body.appendChild(vantaScript);
//     };

//     document.body.appendChild(threeScript);

//     return () => {
//       if (vantaEffectRef.current) vantaEffectRef.current.destroy();
//     };
//   }, []);

//   return (
//     <div className="min-h-screen bg-[#0a0e27] text-white overflow-hidden">
//       {/* Animated gradient background */}
//       <div className="fixed inset-0 z-0">
//         <div className="absolute inset-0 bg-gradient-to-br from-[#0a0e27] via-[#1a1534] to-[#2d1b3d]"></div>
//         <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl animate-pulse"></div>
//         <div
//           className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl animate-pulse"
//           style={{ animationDelay: "1s" }}
//         ></div>
//         <div
//           className="absolute top-1/2 left-1/2 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl animate-pulse"
//           style={{ animationDelay: "2s" }}
//         ></div>
//       </div>



//       {/* Hero Section */}
//       <section
//         ref={vantaRef}
//         className="w-screen h-screen"
//       >
//        <div className="w-full h-full relative z-10 grid md:grid-cols-2 gap-12 items-center px-8 py-2">
//          <div>
//           <p className="text-blue-400 text-sm mb-4 tracking-wider">
//             SIMPLIFY YOUR SECURITY
//           </p>
//           <h1 className="text-6xl font-bold mb-6 leading-tight">SkyFort</h1>
//           <p className="text-4xl font-light mb-4 text-gray-300">
//             The first cloud-firewall, built for WPFL
//           </p>
//           <p className="text-gray-400 mb-8 leading-relaxed">
//             SkyFort is simple and easy to use, no IT department required. Your
//             team is automatically secured by Fortress as they work from anywhere
//             in the world.
//           </p>
//           <div className="flex space-x-4">
//             <button className="bg-blue-600 hover:bg-blue-700 px-8 py-3 rounded-full font-semibold transition">
//               SIGN UP NOW
//             </button>
//             <button className="border border-gray-600 hover:border-blue-400 px-8 py-3 rounded-full font-semibold transition">
//               READ MORE
//             </button>
//           </div>
//         </div>

//         {/* <div className="relative h-[500px] flex items-center justify-center">
//           <div className="relative w-full max-w-xl">
           

//             <div
//               className="relative bg-gradient-to-br from-gray-900/90 to-blue-900/40 backdrop-blur-xl rounded-2xl border border-blue-500/30 shadow-2xl overflow-hidden"
//               style={{
//                 boxShadow:
//                   "0 25px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(59, 130, 246, 0.2)",
//               }}
//             >
//               <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 border-b border-blue-500/30 px-6 py-4 flex items-center justify-between">
//                 <div className="flex items-center space-x-3">
//                   <div className="w-3 h-3 bg-red-500 rounded-full"></div>
//                   <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
//                   <div className="w-3 h-3 bg-green-500 rounded-full"></div>
//                   <span className="ml-4 text-sm font-mono text-gray-400">
//                     Security Monitor
//                   </span>
//                 </div>
//                 <div className="flex items-center space-x-2">
//                   <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
//                   <span className="text-xs text-green-400 font-mono">LIVE</span>
//                 </div>
//               </div>

//               <div className="p-6 space-y-3">
//                 {[
//                   {
//                     type: "Benign",
//                     count: "1,895,314",
//                     color: "from-green-500 to-emerald-500",
//                     bg: "bg-green-500/10",
//                     border: "border-green-500/30",
//                     icon: CheckCircle,
//                   },
//                   {
//                     type: "DoS Hulk Attack",
//                     count: "172,846",
//                     color: "from-red-500 to-orange-500",
//                     bg: "bg-red-500/10",
//                     border: "border-red-500/30",
//                     icon: Shield,
//                   },
//                   {
//                     type: "DDoS Attack",
//                     count: "128,014",
//                     color: "from-orange-500 to-yellow-500",
//                     bg: "bg-orange-500/10",
//                     border: "border-orange-500/30",
//                     icon: Lock,
//                   },
//                   {
//                     type: "DoS GoldenEye Attack",
//                     count: "10,286",
//                     color: "from-yellow-500 to-amber-500",
//                     bg: "bg-yellow-500/10",
//                     border: "border-yellow-500/30",
//                     icon: Shield,
//                   },
//                   {
//                     type: "FTP Brute Force Attack",
//                     count: "5,931",
//                     color: "from-purple-500 to-pink-500",
//                     bg: "bg-purple-500/10",
//                     border: "border-purple-500/30",
//                     icon: Lock,
//                   },
//                 ].map((log, i) => (
//                   <div
//                     key={i}
//                     className={`${log.bg} ${log.border} border rounded-xl p-4 flex items-center justify-between hover:scale-[1.02] transition-all duration-300 group`}
//                     style={{
//                       animationDelay: `${i * 0.1}s`,
//                       animation: "slideIn 0.5s ease-out forwards",
//                       opacity: 0,
//                     }}
//                   >
//                     <div className="flex items-center space-x-4 flex-1">
//                       <div
//                         className={`w-10 h-10 rounded-lg bg-gradient-to-br ${log.color} flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform`}
//                       >
//                         <log.icon className="w-5 h-5 text-white" />
//                       </div>
//                       <div className="flex-1">
//                         <div className="text-sm font-semibold text-white mb-1">
//                           {log.type}
//                         </div>
//                         <div className="text-xs text-gray-400 font-mono">
//                           Events detected
//                         </div>
//                       </div>
//                     </div>
//                     <div className="text-right">
//                       <div
//                         className={`text-xl font-bold bg-gradient-to-r ${log.color} bg-clip-text text-transparent`}
//                       >
//                         {log.count}
//                       </div>
//                       <div className="text-xs text-gray-500 font-mono">
//                         logged
//                       </div>
//                     </div>
//                   </div>
//                 ))}
//               </div>

//               <div className="bg-gradient-to-r from-blue-950/50 to-purple-950/50 border-t border-blue-500/20 px-6 py-3 flex items-center justify-between">
//                 <div className="flex items-center space-x-4 text-xs text-gray-400">
//                   <span className="flex items-center space-x-2">
//                     <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"></div>
//                     <span>Monitoring Active</span>
//                   </span>
//                   <span className="text-gray-600">|</span>
//                   <span>2,212,391 Total Events</span>
//                 </div>
//                 <span className="text-xs text-gray-500 font-mono">
//                   Updated 2s ago
//                 </span>
//               </div>
//             </div>

           

//             <div className="absolute top-1/4 -left-12 w-24 h-24 bg-blue-500/20 rounded-full blur-2xl animate-pulse"></div>
//             <div
//               className="absolute bottom-1/4 -right-12 w-32 h-32 bg-purple-500/20 rounded-full blur-2xl animate-pulse"
//               style={{ animationDelay: "1.5s" }}
//             ></div>
//           </div>

//           <style jsx>{`
//             @keyframes slideIn {
//               from {
//                 opacity: 0;
//                 transform: translateX(-20px);
//               }
//               to {
//                 opacity: 1;
//                 transform: translateX(0);
//               }
//             }
//           `}</style>
//         </div> */}
//        </div>
//       </section>

//       {/* Customer Logos */}
//       <section className="relative z-10 max-w-7xl mx-auto px-8 py-16">
//         <p className="text-center text-gray-400 mb-12">
//           Trusted by 350+ happy customers
//         </p>
//         <div className="grid grid-cols-3 md:grid-cols-7 gap-8 items-center opacity-40">
//           {[
//             "CGV",
//             "Hmall",
//             "pghm",
//             "LOTTE",
//             "NAVER",
//             "coupang",
//             "SK telecom",
//             "saramth",
//             "coupang",
//             "MBC",
//             "CGV",
//             "KBS",
//           ].map((company, i) => (
//             <div
//               key={i}
//               className="text-center text-gray-500 font-bold text-xl"
//             >
//               {company}
//             </div>
//           ))}
//         </div>
//       </section>

//       {/* Why SkyFort Section */}
//       <section className="relative z-10 max-w-7xl mx-auto px-8 py-20">
//         <div className="text-center mb-16">
//           <p className="text-blue-400 text-sm mb-4 tracking-wider">
//             POWERFULL FEATURES
//           </p>
//           <h2 className="text-4xl font-bold mb-4">Why SkyFort?</h2>
//           <p className="text-gray-400 max-w-2xl mx-auto">
//             SkyFort is the first and only easy to use cloud firewall for remote
//             teams.
//           </p>
//         </div>

//         <div className="grid md:grid-cols-3 gap-8">
//           <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/20 backdrop-blur-sm border border-blue-800/30 rounded-2xl p-8 hover:border-blue-600/50 transition">
//             <div className="bg-gradient-to-br from-orange-500 to-red-500 w-12 h-12 rounded-xl flex items-center justify-center mb-6">
//               <Shield className="w-6 h-6 text-white" />
//             </div>
//             <h3 className="text-xl font-bold mb-4">Reliable protection</h3>
//             <p className="text-gray-400 leading-relaxed">
//               We manage your network traffic, keeping your site protected
//               smoothly.
//             </p>
//           </div>

//           <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/20 backdrop-blur-sm border border-blue-800/30 rounded-2xl p-8 hover:border-blue-600/50 transition">
//             <div className="bg-gradient-to-br from-blue-500 to-cyan-500 w-12 h-12 rounded-xl flex items-center justify-center mb-6">
//               <Zap className="w-6 h-6 text-white" />
//             </div>
//             <h3 className="text-xl font-bold mb-4">Easy to set up</h3>
//             <p className="text-gray-400 leading-relaxed">
//               Your team's data is secured and protected instantly.
//             </p>
//           </div>

//           <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/20 backdrop-blur-sm border border-blue-800/30 rounded-2xl p-8 hover:border-blue-600/50 transition">
//             <div className="bg-gradient-to-br from-orange-500 to-yellow-500 w-12 h-12 rounded-xl flex items-center justify-center mb-6">
//               <CheckCircle className="w-6 h-6 text-white" />
//             </div>
//             <h3 className="text-xl font-bold mb-4">Virus protection</h3>
//             <p className="text-gray-400 leading-relaxed">
//               All web traffic is intelligently screened for malware and spyware.
//             </p>
//           </div>
//         </div>
//       </section>

//       {/* Features Grid */}
//       <section className="relative z-10 max-w-7xl mx-auto px-8 py-12">
//         <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
//           {[
//             {
//               icon: Lock,
//               title: "Secure access",
//               desc: "Control who has access to your network",
//             },
//             {
//               icon: Shield,
//               title: "Network level protection",
//               desc: "Protect against DDoS attacks",
//             },
//             {
//               icon: Cloud,
//               title: "Cloud-native",
//               desc: "Built for modern cloud infrastructure",
//             },
//             {
//               icon: Users,
//               title: "Team collaboration",
//               desc: "Easy team management and access",
//             },
//             {
//               icon: Zap,
//               title: "Lightning fast",
//               desc: "Minimal latency, maximum speed",
//             },
//             {
//               icon: CheckCircle,
//               title: "Always available",
//               desc: "99.9% uptime guarantee",
//             },
//             { icon: Lock, title: "Encrypted", desc: "End-to-end encryption" },
//             {
//               icon: Shield,
//               title: "24/7 monitoring",
//               desc: "Round the clock security",
//             },
//           ].map((feature, i) => (
//             <div
//               key={i}
//               className="bg-gradient-to-br from-blue-900/20 to-purple-900/10 backdrop-blur-sm border border-blue-800/20 rounded-xl p-6 hover:border-blue-600/40 transition"
//             >
//               <feature.icon className="w-8 h-8 text-blue-400 mb-4" />
//               <h4 className="font-semibold mb-2">{feature.title}</h4>
//               <p className="text-sm text-gray-400">{feature.desc}</p>
//             </div>
//           ))}
//         </div>
//       </section>

//       {/* CTA Section */}
//       <section className="relative z-10 max-w-4xl mx-auto px-8 py-20 text-center">
//         <h2 className="text-4xl font-bold mb-6">Ready to get started?</h2>
//         <p className="text-gray-400 mb-8 text-lg">
//           Join 350+ companies protecting their teams with SkyFort
//         </p>
//         <button className="bg-blue-600 hover:bg-blue-700 px-10 py-4 rounded-full text-lg font-semibold transition shadow-lg hover:shadow-blue-500/50">
//           Get Started Now
//         </button>
//       </section>

//       {/* Footer */}
//       <footer className="relative z-10 border-t border-gray-800 mt-20">
//         <div className="max-w-7xl mx-auto px-8 py-12">
//           <div className="flex flex-col md:flex-row justify-between items-center">
//             <div className="flex items-center space-x-2 mb-4 md:mb-0">
//               <Shield className="w-6 h-6 text-blue-400" />
//               <span className="text-xl font-bold">SkyFort</span>
//             </div>
//             <div className="text-gray-400 text-sm">
//               © 2024 SkyFort. All rights reserved.
//             </div>
//           </div>
//         </div>
//       </footer>
//     </div>
//   );
// }










import React, { useEffect, useRef } from "react";
import { Shield, Lock, Cloud, Zap, CheckCircle, Users } from "lucide-react";

export default function SkyFortLanding() {
  const vantaRef = useRef<HTMLDivElement | null>(null);
  const vantaEffectRef = useRef<any>(null);

  useEffect(() => {
    // Load THREE
    const threeScript = document.createElement("script");
    threeScript.src =
      "https://cdnjs.cloudflare.com/ajax/libs/three.js/r121/three.min.js";

    threeScript.onload = () => {
      const vantaScript = document.createElement("script");
      vantaScript.src =
        "https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.globe.min.js";

      vantaScript.onload = () => {
        if (!vantaEffectRef.current && (window as any).VANTA && vantaRef.current) {
          vantaEffectRef.current = (window as any).VANTA.GLOBE({
            el: vantaRef.current,
            mouseControls: true,
            touchControls: true,
            minHeight: 200,
            minWidth: 200,
            scale: 1.0,
            scaleMobile: 1.0,
            color: 0x543b54,
            backgroundColor: 0x1f1f1f,
          });
        }
      };

      document.body.appendChild(vantaScript);
    };

    document.body.appendChild(threeScript);

    return () => {
      if (vantaEffectRef.current) vantaEffectRef.current.destroy();
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#1f1f1f] text-white overflow-hidden">

      {/* Subtle Gradient Glows */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-[#1f1f1f] via-[#2a2230] to-[#3a2f42]" />
        <div className="absolute top-1/3 left-1/4 w-80 h-80 bg-[#543b54]/25 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/3 right-1/3 w-80 h-80 bg-[#6b4b6c]/25 rounded-full blur-3xl animate-pulse" />
      </div>

      {/* Hero Section with VANTA */}
      <section ref={vantaRef} className="w-screen h-screen">
        <div className="w-full h-full relative z-10 grid md:grid-cols-2 gap-12 items-center px-8">
          
          {/* Left */}
          <div>
            <p className="text-[#c5a4c5] text-sm mb-4 tracking-wider">
              SIMPLIFY YOUR SECURITY
            </p>

            <h1 className="text-6xl font-bold mb-6 leading-tight text-[#e5dce7]">
              SkyFort
            </h1>

            <p className="text-3xl font-light mb-4 text-[#c7b5ca]">
              The first cloud-firewall, built for WPFL
            </p>

            <p className="text-gray-400 mb-8 leading-relaxed">
              SkyFort is simple and easy to use — no IT department required.
              Your team stays secured from anywhere in the world.
            </p>

            <div className="flex space-x-4">
              <button className="bg-[#543b54] hover:bg-[#6d4d6e] px-8 py-3 rounded-full font-semibold transition">
                SIGN UP NOW
              </button>
              <button className="border border-[#543b54]/60 hover:border-[#6d4d6e] px-8 py-3 rounded-full font-semibold transition">
                READ MORE
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Trusted Logos */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 py-16">
        <p className="text-center text-gray-400 mb-12">
          Trusted by 350+ happy customers
        </p>

        <div className="grid grid-cols-3 md:grid-cols-7 gap-8 items-center opacity-30">
          {[
            "CGV", "Hmall", "pghm", "LOTTE", "NAVER",
            "coupang", "SK telecom", "saramth", "MBC", "KBS",
          ].map((company, i) => (
            <div key={i} className="text-center text-gray-500 font-bold text-xl">
              {company}
            </div>
          ))}
        </div>
      </section>

      {/* Why SkyFort */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 py-20">
        <div className="text-center mb-16">
          <p className="text-[#c5a4c5] text-sm mb-4 tracking-wider">
            POWERFUL FEATURES
          </p>
          <h2 className="text-4xl font-bold mb-4">Why SkyFort?</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            SkyFort is the first cloud firewall designed for remote teams — simple, modern, secure.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {[
            {
              icon: Shield,
              title: "Reliable protection",
              desc: "We intelligently manage your network traffic."
            },
            {
              icon: Zap,
              title: "Easy to set up",
              desc: "Your team gets instant protection.",
            },
            {
              icon: CheckCircle,
              title: "Virus protection",
              desc: "Traffic is scanned for malware threats.",
            },
          ].map((box, i) => (
            <div
              key={i}
              className="bg-[#543b54]/10 border border-[#543b54]/30 rounded-2xl p-8 backdrop-blur-sm hover:border-[#7a5980]/50 transition"
            >
              <div className="bg-gradient-to-br from-[#543b54] to-[#6d4d6e] w-12 h-12 rounded-xl flex items-center justify-center mb-6">
                <box.icon className="w-6 h-6 text-white" />
              </div>

              <h3 className="text-xl font-bold mb-4">{box.title}</h3>
              <p className="text-gray-400">{box.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features Grid */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 py-12">
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { icon: Lock, title: "Secure access", desc: "Control access permissions" },
            { icon: Shield, title: "Network protection", desc: "Defense against DDoS" },
            { icon: Cloud, title: "Cloud-native", desc: "Optimized for modern infra" },
            { icon: Users, title: "Team collaboration", desc: "Simple team management" },
            { icon: Zap, title: "Lightning fast", desc: "Minimum latency" },
            { icon: CheckCircle, title: "Always available", desc: "99.9% uptime" },
            { icon: Lock, title: "Encrypted", desc: "End-to-end security" },
            { icon: Shield, title: "24/7 monitoring", desc: "Continuous protection" },
          ].map((f, i) => (
            <div
              key={i}
              className="bg-[#543b54]/10 border border-[#543b54]/20 rounded-xl p-6 backdrop-blur-sm hover:border-[#7a5980]/40 transition"
            >
              <f.icon className="w-8 h-8 text-[#c5a4c5] mb-4" />
              <h4 className="font-semibold mb-2">{f.title}</h4>
              <p className="text-sm text-gray-400">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="relative z-10 max-w-4xl mx-auto px-8 py-20 text-center">
        <h2 className="text-4xl font-bold mb-6">Ready to get started?</h2>
        <p className="text-gray-400 mb-8 text-lg">
          Join 350+ companies securing their teams with SkyFort.
        </p>
        <button className="bg-[#543b54] hover:bg-[#6d4d6e] px-10 py-4 rounded-full text-lg font-semibold transition">
          Get Started Now
        </button>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-[#543b54]/20 mt-20">
        <div className="max-w-7xl mx-auto px-8 py-12 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Shield className="w-6 h-6 text-[#c5a4c5]" />
            <span className="text-xl font-bold">SkyFort</span>
          </div>
          <div className="text-gray-400 text-sm">
            © 2024 SkyFort. All rights reserved.
          </div>
        </div>
      </footer>

    </div>
  );
}
