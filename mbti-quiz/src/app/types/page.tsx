"use client";

import Link from "next/link";
import { allPersonalityTypes } from "@/data/personalities";
import { useState } from "react";

export default function PersonalityTypesPage() {
    return (
        <main className="min-h-screen bg-[#F5F2E8] text-[#2C2C2C] pb-12">
            {/* Header / Nav */}
            <div className="bg-[#8B1538] text-white py-6 px-6 md:px-12 shadow-md mb-8">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                    <div>
                        <Link
                            href="/"
                            className="inline-flex items-center text-white/80 hover:text-white transition-colors mb-2 text-sm font-medium"
                        >
                            ← Back to Home
                        </Link>
                        <h1 className="text-3xl md:text-4xl font-bold">16 Personality Types</h1>
                    </div>
                    <p className="text-white/80 max-w-md text-sm md:text-base">
                        Explore the detailed characteristics of each personality type to understand their unique strengths and preferences.
                    </p>
                </div>
            </div>

            <div className="max-w-7xl mx-auto px-4 md:px-8">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {allPersonalityTypes.map((type) => (
                        <div
                            key={type.code}
                            className="bg-white rounded-xl shadow-md border-t-4 transition-all duration-300 hover:shadow-lg hover:-translate-y-1"
                            style={{ borderColor: type.color }}
                        >
                            <div className="p-6 h-full flex flex-col">
                                <div className="flex justify-between items-start mb-4">
                                    <div>
                                        <h2 className="text-2xl font-bold text-[#2C2C2C] flex items-center gap-2">
                                            {type.code}
                                        </h2>
                                        <p className="text-[#8B1538] font-medium text-lg">{type.nickname}</p>
                                    </div>
                                    <div
                                        className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-lg"
                                        style={{ backgroundColor: type.color }}
                                    >
                                        →
                                    </div>
                                </div>

                                <p className="text-gray-600 leading-relaxed line-clamp-3 mb-6 flex-grow">
                                    {type.description}
                                </p>

                                <Link
                                    href={`/types/${type.code}`}
                                    className="w-full mt-auto py-2 text-center text-sm font-bold text-[#4A7C7C] hover:text-[#2C2C2C] hover:bg-gray-50 rounded-lg transition-colors border border-transparent hover:border-gray-200 block"
                                >
                                    Read Full Profile
                                </Link>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </main>
    );
}
