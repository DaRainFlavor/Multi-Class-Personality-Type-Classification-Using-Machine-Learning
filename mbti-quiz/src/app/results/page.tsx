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

        // Use requestAnimationFrame to avoid synchronous state update warning in effect
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

    return (
        <main className="min-h-screen bg-[#F5F2E8] text-[#2C2C2C] py-12 px-4">
            {/* Decorative shapes */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-20 left-10 w-40 h-40 bg-[#C4A52D] rounded-full opacity-30" />
                <div className="absolute top-60 right-20 w-32 h-32 bg-[#4A7C7C] rounded-full opacity-30" />
                <div className="absolute bottom-40 left-1/4 w-48 h-48 bg-[#8B1538] rounded-full opacity-20" />
            </div>

            <div className="relative z-10 max-w-4xl mx-auto">
                {/* Header */}
                <div className="text-center mb-12">
                    <p className="text-[#4A7C7C] text-lg font-medium mb-2">Your personality type is</p>
                    <h1 className="text-7xl md:text-9xl font-bold mb-4 tracking-wider text-[#8B1538]">
                        {personality.code}
                    </h1>
                    <h2 className="text-3xl md:text-4xl font-medium text-[#2C2C2C] mb-2">
                        {personality.name}
                    </h2>
                    <p className="text-xl text-[#C4A52D] font-medium">{personality.nickname}</p>
                </div>

                {/* Confidence Score */}
                <div className="bg-white rounded-3xl p-6 border-2 border-[#C4A52D] shadow-lg mb-8">
                    <div className="flex items-center justify-between mb-4">
                        <span className="text-[#4A7C7C] font-medium">Confidence Score</span>
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

                {/* Description */}
                <div className="bg-white rounded-3xl p-8 border-2 border-[#4A7C7C] shadow-lg mb-8">
                    <h3 className="text-xl font-bold mb-4 text-[#8B1538]">About Your Type</h3>
                    <p className="text-lg text-[#2C2C2C] leading-relaxed">
                        {personality.description}
                    </p>
                </div>

                {/* Traits and Strengths */}
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                    <div className="bg-white rounded-3xl p-6 border-2 border-[#C4A52D] shadow-lg">
                        <h3 className="text-lg font-bold mb-4 text-[#C4A52D]">Key Traits</h3>
                        <div className="flex flex-wrap gap-2">
                            {personality.traits.map((trait) => (
                                <span
                                    key={trait}
                                    className="px-4 py-2 rounded-full text-sm font-medium bg-[#8B1538] text-white"
                                >
                                    {trait}
                                </span>
                            ))}
                        </div>
                    </div>

                    <div className="bg-white rounded-3xl p-6 border-2 border-[#4A7C7C] shadow-lg">
                        <h3 className="text-lg font-bold mb-4 text-[#4A7C7C]">Strengths</h3>
                        <div className="flex flex-wrap gap-2">
                            {personality.strengths.map((strength) => (
                                <span
                                    key={strength}
                                    className="px-4 py-2 rounded-full text-sm font-medium bg-[#4A7C7C] text-white"
                                >
                                    {strength}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Probability Distribution */}
                <div className="bg-white rounded-3xl p-6 border-2 border-[#8B1538] shadow-lg mb-8">
                    <h3 className="text-lg font-bold mb-4 text-[#8B1538]">Top 5 Type Matches</h3>
                    <div className="space-y-3">
                        {topProbabilities.map(([type, prob], index) => (
                            <div key={type} className="flex items-center gap-4">
                                <span
                                    className={`text-sm font-mono w-16 font-bold ${index === 0 ? "text-[#8B1538]" : "text-[#4A7C7C]"
                                        }`}
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
