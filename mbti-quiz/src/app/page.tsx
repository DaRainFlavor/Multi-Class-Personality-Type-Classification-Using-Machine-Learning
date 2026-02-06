"use client";

import Link from "next/link";
import { useState } from "react";

export default function Home() {
  const [showModeSelect, setShowModeSelect] = useState(false);

  return (
    <main className="min-h-screen bg-[#F5F2E8] text-[#2C2C2C] flex flex-col items-center justify-center p-4 overflow-hidden">
      {/* Decorative shapes */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-10 left-10 w-20 h-20 md:w-32 md:h-32 bg-[#C4A52D] rounded-full opacity-60" />
        <div className="absolute top-40 right-10 md:right-20 w-16 h-16 md:w-24 md:h-24 bg-[#4A7C7C] rounded-full opacity-60" />
        <div className="absolute bottom-20 left-4 md:left-1/4 w-24 h-24 md:w-40 md:h-40 bg-[#8B1538] rounded-full opacity-40" />
        <div className="absolute bottom-40 right-4 md:right-1/3 w-16 h-16 md:w-20 md:h-20 bg-[#C41E3A] rounded-full opacity-50" />
      </div>

      {/* Main content */}
      <div className="relative z-10 max-w-3xl mx-auto text-center w-full flex-grow flex flex-col justify-center">
        {/* Logo/Title */}
        <div className="mb-6 md:mb-8">
          <h1 className="text-5xl sm:text-6xl md:text-7xl font-bold mb-2 md:mb-4 text-[#8B1538]">
            MBTI Quiz
          </h1>
          <p className="text-lg md:text-2xl text-[#4A7C7C] font-medium">
            Discover Your Personality Type
          </p>
        </div>

        {/* Description Card */}
        <div className="bg-white rounded-2xl md:rounded-3xl p-5 md:p-8 mb-6 md:mb-8 shadow-lg border-2 border-[#C4A52D] flex flex-col items-center text-center">
          <p className="text-lg md:text-xl text-[#2C2C2C] leading-relaxed max-w-2xl mb-6">
            Take this comprehensive personality assessment based on the Myers-Briggs Type Indicator (MBTI).
          </p>
          <div className="flex flex-row justify-center gap-2 md:gap-4 w-full px-1">
            <Link
              href="/types"
              className="px-3 md:px-6 py-2 rounded-full bg-white border-2 border-[#4A7C7C] text-[#4A7C7C] font-semibold text-sm md:text-base hover:bg-[#4A7C7C] hover:text-white transition-all duration-300"
            >
              Personality Types
            </Link>
            <Link
              href="/references"
              className="px-3 md:px-6 py-2 rounded-full bg-white border-2 border-[#C4A52D] text-[#C4A52D] font-semibold text-sm md:text-base hover:bg-[#C4A52D] hover:text-white transition-all duration-300"
            >
              References
            </Link>
          </div>
        </div>

        {/* Mode Selection */}
        <div className="flex flex-col items-center gap-4">
          <Link
            href="/quiz-mode"
            className="inline-block px-8 py-4 md:px-12 md:py-5 text-lg md:text-xl font-bold rounded-full bg-[#8B1538] text-white hover:bg-[#6B1028] transition-all duration-300 shadow-lg hover:scale-105 transform active:scale-95"
          >
            Start Quiz
          </Link>
        </div>
      </div>

      {/* Footer note */}
      <p className="w-full text-center py-4 text-xs md:text-sm text-[#4A7C7C] opacity-80 relative z-10">
        Powered by XGBoost Machine Learning Model
      </p>
    </main>
  );
}
