"use client";

import Link from "next/link";
import { getPersonalityType } from "@/data/personalities";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { PersonalityType } from "@/data/personalities";

export default function PersonalityTypePage() {
    const params = useParams();
    const router = useRouter();
    const [type, setType] = useState<PersonalityType | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (params.code) {
            const code = Array.isArray(params.code) ? params.code[0] : params.code;
            const personality = getPersonalityType(code);
            if (personality) {
                setType(personality);
            } else {
                router.push("/types");
            }
            setLoading(false);
        }
    }, [params.code, router]);

    if (loading) {
        return (
            <div className="min-h-screen bg-[#F5F2E8] flex items-center justify-center">
                <div className="text-[#8B1538] text-xl font-bold">Loading...</div>
            </div>
        );
    }

    if (!type) return null;

    return (
        <main className="min-h-screen bg-[#F5F2E8] text-[#2C2C2C] pb-12">
            {/* Header / Nav */}
            <div className="bg-[#8B1538] text-white py-6 px-6 md:px-12 shadow-md mb-8">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                    <div>
                        <Link
                            href="/types"
                            className="inline-flex items-center text-white/80 hover:text-white transition-colors mb-2 text-sm font-medium"
                        >
                            ← Back to Personality Types
                        </Link>
                        <h1 className="text-3xl md:text-5xl font-bold flex items-center gap-3">
                            {type.code}
                        </h1>
                    </div>
                    <p className="text-white/80 text-lg md:text-xl font-medium">
                        {type.nickname}
                    </p>
                </div>
            </div>

            <div className="max-w-7xl mx-auto px-4 md:px-8">
                <div
                    className="bg-white rounded-2xl shadow-xl border-t-8 overflow-hidden"
                    style={{ borderColor: type.color }}
                >
                    <div className="p-6 md:p-10">
                        <p className="text-lg md:text-xl text-gray-700 leading-relaxed mb-8">
                            {type.description}
                        </p>

                        <div className="space-y-8 animate-fadeIn">
                            {/* Key Traits Grid */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-[#F5F2E8] p-6 rounded-xl">
                                <div>
                                    <h3 className="font-bold text-[#4A7C7C] mb-2 uppercase text-xs tracking-wide">Most Value</h3>
                                    <p className="font-bold text-lg text-[#2C2C2C]">{type.mostValue}</p>
                                </div>
                                <div>
                                    <h3 className="font-bold text-[#4A7C7C] mb-2 uppercase text-xs tracking-wide">Appear to Others</h3>
                                    <p className="font-bold text-lg text-[#2C2C2C]">{type.appearToOthers}</p>
                                </div>
                            </div>

                            {/* Characteristics */}
                            <div>
                                <h3 className="font-bold text-[#2C2C2C] mb-4 flex items-center gap-2 text-xl">
                                    <span style={{ color: type.color }}>★</span> Characteristics
                                </h3>
                                <div className="flex flex-wrap gap-2 md:gap-3">
                                    {type.characteristics.map((c, i) => (
                                        <span key={i} className="bg-gray-100 text-gray-700 px-4 py-2 rounded-full text-base font-medium">
                                            {c}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            {/* Majors & Careers */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12 pt-4">
                                <div>
                                    <h3 className="font-bold text-[#2C2C2C] mb-4 border-b pb-2 text-lg">Typical Majors</h3>
                                    <ul className="text-gray-600 space-y-2 max-h-96 overflow-y-auto custom-scrollbar pr-2">
                                        {type.majors.map((m, i) => (
                                            <li key={i} className="flex items-start gap-2">
                                                <span className="text-[#C4A52D] mt-1.5 text-xs">●</span>
                                                {m}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div>
                                    <h3 className="font-bold text-[#2C2C2C] mb-4 border-b pb-2 text-lg">Related Careers</h3>
                                    <ul className="text-gray-600 space-y-2 max-h-96 overflow-y-auto custom-scrollbar pr-2">
                                        {type.careers.map((c, i) => (
                                            <li key={i} className="flex items-start gap-2">
                                                <span className="text-[#C4A52D] mt-1.5 text-xs">●</span>
                                                {c}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}
