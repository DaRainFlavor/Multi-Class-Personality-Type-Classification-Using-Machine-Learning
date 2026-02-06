"use client";

import Link from "next/link";

export default function QuizModePage() {
    return (
        <main className="min-h-screen bg-[#F5F2E8] text-[#2C2C2C] flex flex-col">
            {/* Header / Nav */}
            <div className="bg-[#8B1538] text-white py-3 px-6 shadow-md z-10">
                <div className="max-w-7xl mx-auto flex items-center justify-between">
                    <Link
                        href="/"
                        className="inline-flex items-center text-white/80 hover:text-white transition-colors text-sm font-medium"
                    >
                        ← Back to Home
                    </Link>
                </div>
            </div>

            <div className="flex-grow flex flex-col items-center justify-center p-4">
                <div className="max-w-4xl w-full text-center h-full flex flex-col justify-center">
                    <h1 className="text-3xl md:text-5xl font-bold text-[#8B1538] mb-2 mt-[-1rem]">
                        Choose Your Path
                    </h1>

                    <p className="text-sm md:text-lg text-[#4A7C7C] mb-6 max-w-2xl mx-auto leading-relaxed px-4">
                        Select the assessment that best fits your schedule. Both options are designed to reveal your true personality type.
                    </p>

                    <div className="flex flex-col md:flex-row justify-center gap-4 max-w-4xl mx-auto w-full">
                        {/* Full Assessment Card */}
                        <Link
                            href="/quiz?mode=full"
                            className="group relative bg-white rounded-2xl md:rounded-3xl p-5 shadow-lg border-2 border-transparent hover:border-[#8B1538] transition-all duration-300 hover:shadow-2xl hover:-translate-y-1 flex flex-col items-center flex-1"
                        >
                            <div className="w-10 h-10 md:w-16 md:h-16 bg-[#FFF5F7] rounded-full flex items-center justify-center mb-2 group-hover:scale-110 transition-transform duration-300">
                                <span className="text-xl md:text-4xl">📋</span>
                            </div>
                            <h2 className="text-lg md:text-2xl font-bold text-[#8B1538] mb-1 group-hover:text-[#6B1028] transition-colors leading-tight">
                                Full Assessment
                            </h2>
                            <ul className="text-gray-600 mb-3 space-y-1 text-xs md:text-sm">
                                <li className="flex items-center gap-1 justify-center">
                                    <span className="text-[#8B1538] font-bold">60</span> Questions
                                </li>
                                <li className="flex items-center gap-1 justify-center">
                                    <span className="text-[#8B1538] font-bold">~10</span> Minutes
                                </li>
                                <li className="text-[#C4A52D] font-medium mt-1">Most Accurate</li>
                            </ul>
                            <span className="mt-auto inline-block px-6 py-2 rounded-full bg-[#8B1538] text-white text-xs md:text-sm font-bold group-hover:bg-[#6B1028] transition-colors w-full">
                                Start Full
                            </span>
                        </Link>

                        {/* Quick Assessment Card */}
                        <Link
                            href="/quiz?mode=short"
                            className="group relative bg-white rounded-2xl md:rounded-3xl p-5 shadow-lg border-2 border-transparent hover:border-[#4A7C7C] transition-all duration-300 hover:shadow-2xl hover:-translate-y-1 flex flex-col items-center flex-1"
                        >
                            <div className="w-10 h-10 md:w-16 md:h-16 bg-[#F0F7F7] rounded-full flex items-center justify-center mb-2 group-hover:scale-110 transition-transform duration-300">
                                <span className="text-xl md:text-4xl">⚡</span>
                            </div>
                            <h2 className="text-lg md:text-2xl font-bold text-[#4A7C7C] mb-1 group-hover:text-[#3A6060] transition-colors leading-tight">
                                Quick Assessment
                            </h2>
                            <ul className="text-gray-600 mb-3 space-y-1 text-xs md:text-sm">
                                <li className="flex items-center gap-1 justify-center">
                                    <span className="text-[#4A7C7C] font-bold">35</span> Questions
                                </li>
                                <li className="flex items-center gap-1 justify-center">
                                    <span className="text-[#4A7C7C] font-bold">~5</span> Minutes
                                </li>
                                <li className="text-[#C4A52D] font-medium mt-1">97% Accuracy</li>
                            </ul>
                            <span className="mt-auto inline-block px-6 py-2 rounded-full bg-[#4A7C7C] text-white text-xs md:text-sm font-bold group-hover:bg-[#3A6060] transition-colors w-full">
                                Start Quick
                            </span>
                        </Link>
                    </div>
                </div>
            </div>
        </main>
    );
}
