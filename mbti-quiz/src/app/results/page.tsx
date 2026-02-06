"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { getPersonalityType, PersonalityType } from "@/data/personalities";

interface QuizResult {
    predicted_type: string;
    confidence: number;
    probabilities: Record<string, number>;
}

// Collapsible Section Component
function CollapsibleSection({ title, children, defaultOpen = false, color = "#8B1538" }: {
    title: string;
    children: React.ReactNode;
    defaultOpen?: boolean;
    color?: string;
}) {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div className="border-2 rounded-2xl overflow-hidden" style={{ borderColor: color }}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full px-6 py-4 flex items-center justify-between bg-white hover:bg-[#F5F2E8] transition-colors"
            >
                <h3 className="text-lg font-bold" style={{ color }}>{title}</h3>
                <svg
                    className={`w-5 h-5 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke={color}
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>
            <div
                className={`transition-all duration-300 ease-in-out overflow-hidden ${isOpen ? 'max-h-[2000px] opacity-100' : 'max-h-0 opacity-0'
                    }`}
            >
                <div className="px-6 py-4 bg-white border-t" style={{ borderColor: color }}>
                    {children}
                </div>
            </div>
        </div>
    );
}

export default function ResultsPage() {
    const router = useRouter();
    const [result, setResult] = useState<QuizResult | null>(null);
    const [personality, setPersonality] = useState<PersonalityType | null>(null);

    useEffect(() => {
        const storedResult = sessionStorage.getItem("quizResult");
        if (!storedResult) {
            router.push("/");
            return;
        }

        const parsedResult = JSON.parse(storedResult) as QuizResult;

        requestAnimationFrame(() => {
            setResult(parsedResult);
            const personalityData = getPersonalityType(parsedResult.predicted_type);
            setPersonality(personalityData || null);
        });
    }, [router]);

    if (!result || !personality) {
        return (
            <main className="min-h-screen bg-[#F5F2E8] text-[#2C2C2C] flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin w-12 h-12 border-4 border-[#8B1538] border-t-transparent rounded-full mx-auto mb-4" />
                    <p className="text-[#4A7C7C] font-medium">Loading your results...</p>
                </div>
            </main>
        );
    }

    const topProbabilities = Object.entries(result.probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5);

    const calculateTraitScores = (probs: Record<string, number>) => {
        let i = 0, e = 0, n = 0, s = 0, t = 0, f = 0, j = 0, p = 0;

        Object.entries(probs).forEach(([type, prob]) => {
            if (type.includes('I')) i += prob; else e += prob;
            if (type.includes('N')) n += prob; else s += prob;
            if (type.includes('T')) t += prob; else f += prob;
            if (type.includes('J')) j += prob; else p += prob;
        });

        return [
            { left: 'Introverted', right: 'Extraverted', leftScore: i, rightScore: e, leftChar: 'I', rightChar: 'E' },
            { left: 'Intuitive', right: 'Observant', leftScore: n, rightScore: s, leftChar: 'N', rightChar: 'S' },
            { left: 'Thinking', right: 'Feeling', leftScore: t, rightScore: f, leftChar: 'T', rightChar: 'F' },
            { left: 'Judging', right: 'Prospecting', leftScore: j, rightScore: p, leftChar: 'J', rightChar: 'P' },
        ];
    };

    const traitScores = calculateTraitScores(result.probabilities);

    return (
        <main className="min-h-screen bg-[#F5F2E8] text-[#2C2C2C] py-12 px-4">
            {/* Decorative shapes */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-20 left-10 w-40 h-40 bg-[#C4A52D] rounded-full opacity-30" />
                <div className="absolute top-60 right-20 w-32 h-32 bg-[#4A7C7C] rounded-full opacity-30" />
                <div className="absolute bottom-40 left-1/4 w-48 h-48 bg-[#8B1538] rounded-full opacity-20" />
            </div>

            <div className="relative z-10 max-w-4xl mx-auto">
                {/* 1. Personality & Alias */}
                <div className="text-center mb-12">
                    <p className="text-[#4A7C7C] text-lg font-medium mb-2">Your personality type is</p>
                    <h1 className="text-7xl md:text-9xl font-bold mb-4 tracking-wider text-[#8B1538]">
                        {personality.code}
                    </h1>
                    <h2 className="text-3xl md:text-4xl font-medium text-[#C4A52D]">
                        {personality.nickname}
                    </h2>
                </div>

                {/* 2. Model Confidence */}
                <div className="bg-white rounded-3xl p-6 border-2 border-[#C4A52D] shadow-lg mb-8">
                    <div className="flex items-center justify-between mb-4">
                        <span className="text-[#4A7C7C] font-medium">Model Confidence</span>
                        <span className="text-2xl font-bold text-[#8B1538]">
                            {(result.confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="w-full h-4 bg-[#E8E5DC] rounded-full overflow-hidden">
                        <div
                            className="h-full rounded-full bg-[#8B1538] transition-all duration-1000"
                            style={{ width: `${result.confidence * 100}%` }}
                        />
                    </div>
                </div>

                {/* 3. About Your Type */}
                <div className="bg-white rounded-3xl p-8 border-2 border-[#4A7C7C] shadow-lg mb-8">
                    <h3 className="text-xl font-bold mb-4 text-[#8B1538]">About Your Type</h3>
                    <p className="text-lg text-[#2C2C2C] leading-relaxed">
                        {personality.description}
                    </p>
                </div>

                {/* 4. What Your Type Means */}
                <div className="bg-white rounded-3xl p-6 md:p-8 border-2 border-[#C4A52D] shadow-lg mb-8">
                    <h3 className="text-xl font-bold mb-6 text-[#C4A52D] text-center">What Your Type Means</h3>
                    <div className="grid md:grid-cols-2 gap-4">
                        {personality.code.split('').map((letter, index) => {
                            const descriptions: Record<string, { title: string; description: string }> = {
                                'E': { title: 'Extroversion', description: 'You gain energy from your outer world.' },
                                'I': { title: 'Introversion', description: 'You gain energy from your inner world.' },
                                'N': { title: 'Intuition', description: 'You value the possible, big picture, and future.' },
                                'S': { title: 'Sensing', description: 'You value the actual, practical, and present.' },
                                'T': { title: 'Thinking', description: 'You base decisions on logic and facts.' },
                                'F': { title: 'Feeling', description: 'You base decisions on what concerns others.' },
                                'J': { title: 'Judging', description: "You're decisive and prefer structure." },
                                'P': { title: 'Perceiving', description: "You're adaptable and prefer flexibility." },
                            };
                            const info = descriptions[letter];
                            if (!info) return null;
                            return (
                                <div key={index} className="flex items-start gap-3 p-4 bg-[#F5F2E8] rounded-xl">
                                    <span className="text-2xl font-bold text-[#8B1538] min-w-[2rem] text-center">{letter}</span>
                                    <div>
                                        <h4 className="font-bold text-[#2C2C2C]">{info.title}</h4>
                                        <p className="text-sm text-[#4A7C7C]">{info.description}</p>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* 5. They Most Value & Appear to Others */}
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                    <div className="bg-white rounded-3xl p-6 border-2 border-[#8B1538] shadow-lg">
                        <h3 className="text-lg font-bold mb-3 text-[#8B1538]">They Most Value</h3>
                        <p className="text-2xl font-bold text-[#2C2C2C]">{personality.mostValue}</p>
                    </div>
                    <div className="bg-white rounded-3xl p-6 border-2 border-[#4A7C7C] shadow-lg">
                        <h3 className="text-lg font-bold mb-3 text-[#4A7C7C]">Appear to Others As</h3>
                        <p className="text-lg font-medium text-[#2C2C2C]">{personality.appearToOthers}</p>
                    </div>
                </div>

                {/* 6. Key Characteristics */}
                <div className="bg-white rounded-3xl p-6 border-2 border-[#C4A52D] shadow-lg mb-8">
                    <h3 className="text-lg font-bold mb-4 text-[#C4A52D]">Key Characteristics</h3>
                    <div className="flex flex-wrap gap-2">
                        {personality.characteristics.map((trait) => (
                            <span
                                key={trait}
                                className="px-4 py-2 rounded-full text-sm font-medium bg-[#8B1538] text-white"
                            >
                                {trait}
                            </span>
                        ))}
                    </div>
                </div>

                {/* 7. Personality Breakdown */}
                <div className="bg-white rounded-3xl p-6 md:p-8 border-2 border-[#8B1538] shadow-lg mb-8">
                    <h3 className="text-xl font-bold mb-6 text-[#8B1538] text-center">Personality Breakdown</h3>
                    <div className="space-y-6">
                        {traitScores.map((trait) => (
                            <div key={trait.left} className="flex flex-col">
                                <div className="flex justify-between mb-2 text-sm md:text-base font-bold text-[#2C2C2C]">
                                    <span className={trait.leftScore >= trait.rightScore ? "text-[#8B1538]" : "text-[#4A7C7C]"}>
                                        {trait.left} ({Math.round(trait.leftScore * 100)}%)
                                    </span>
                                    <span className={trait.rightScore > trait.leftScore ? "text-[#8B1538]" : "text-[#4A7C7C]"}>
                                        {trait.right} ({Math.round(trait.rightScore * 100)}%)
                                    </span>
                                </div>
                                <div className="h-4 bg-[#E8E5DC] rounded-full overflow-hidden flex relative">
                                    <div
                                        className={`h-full transition-all duration-1000 ${trait.leftScore >= trait.rightScore ? "bg-[#8B1538]" : "bg-[#4A7C7C]"}`}
                                        style={{ width: `${trait.leftScore * 100}%` }}
                                    />
                                    <div
                                        className={`h-full transition-all duration-1000 ${trait.rightScore > trait.leftScore ? "bg-[#8B1538]" : "bg-[#4A7C7C]"}`}
                                        style={{ width: `${trait.rightScore * 100}%` }}
                                    />
                                    <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-white/50 z-10" />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* 8. Top 5 Type Matches */}
                <div className="bg-white rounded-3xl p-6 border-2 border-[#8B1538] shadow-lg mb-8">
                    <h3 className="text-lg font-bold mb-4 text-[#8B1538]">Top 5 Type Matches</h3>
                    <div className="space-y-3">
                        {topProbabilities.map(([type, prob], index) => (
                            <div key={type} className="flex items-center gap-4">
                                <span
                                    className={`text-sm font-mono w-16 font-bold ${index === 0 ? "text-[#8B1538]" : "text-[#4A7C7C]"}`}
                                >
                                    {type}
                                </span>
                                <div className="flex-1 h-3 bg-[#E8E5DC] rounded-full overflow-hidden">
                                    <div
                                        className="h-full rounded-full"
                                        style={{
                                            width: `${prob * 100}%`,
                                            backgroundColor: index === 0 ? "#8B1538" : "#C4A52D",
                                        }}
                                    />
                                </div>
                                <span className="text-sm text-[#4A7C7C] font-medium w-16 text-right">
                                    {(prob * 100).toFixed(1)}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* 9 & 10. Collapsible Sections: Majors & Careers */}
                <div className="space-y-4 mb-8">
                    <CollapsibleSection title={`Recommended Majors (${personality.majors.length})`} color="#4A7C7C">
                        <div className="flex flex-wrap gap-2">
                            {personality.majors.map((major) => (
                                <span
                                    key={major}
                                    className="px-3 py-1.5 rounded-full text-sm bg-[#E8E5DC] text-[#2C2C2C] border border-[#4A7C7C]"
                                >
                                    {major}
                                </span>
                            ))}
                        </div>
                    </CollapsibleSection>

                    <CollapsibleSection title={`Potential Careers (${personality.careers.length})`} color="#8B1538">
                        <div className="flex flex-wrap gap-2">
                            {personality.careers.map((career) => (
                                <span
                                    key={career}
                                    className="px-3 py-1.5 rounded-full text-sm bg-[#E8E5DC] text-[#2C2C2C] border border-[#8B1538]"
                                >
                                    {career}
                                </span>
                            ))}
                        </div>
                    </CollapsibleSection>
                </div>

                {/* Actions */}
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Link
                        href="/quiz"
                        className="px-8 py-4 rounded-xl font-bold bg-[#4A7C7C] text-white hover:bg-[#2D5A5A] transition-all text-center"
                    >
                        Retake Quiz
                    </Link>
                    <Link
                        href="/"
                        className="px-8 py-4 rounded-xl font-bold bg-[#8B1538] text-white hover:bg-[#6B1028] transition-all text-center"
                    >
                        Back to Home
                    </Link>
                </div>

                {/* Footer */}
                <p className="text-center text-sm text-[#4A7C7C] mt-12">
                    Results generated using XGBoost Machine Learning Model
                </p>
            </div>
        </main>
    );
}
