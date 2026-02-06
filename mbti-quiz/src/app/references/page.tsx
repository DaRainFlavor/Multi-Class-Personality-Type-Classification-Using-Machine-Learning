"use client";

import Link from "next/link";

export default function ReferencesPage() {
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
                        <h1 className="text-3xl md:text-4xl font-bold">References</h1>
                    </div>
                    <p className="text-white/80 max-w-md text-sm md:text-base">
                        Sources and documentation for the project.
                    </p>
                </div>
            </div>

            <div className="max-w-7xl mx-auto px-4 md:px-8">
                <div className="bg-white rounded-2xl p-6 md:p-10 shadow-lg border-t-4 border-[#C4A52D]">
                    <div className="space-y-8">
                        <div>
                            <h2 className="text-xl md:text-2xl font-bold text-[#4A7C7C] mb-3">
                                Model Dataset
                            </h2>
                            <p className="text-gray-600 mb-2">
                                Synthetic dataset used for training the machine learning model:
                            </p>
                            <a
                                href="https://www.kaggle.com/datasets/anshulmehtakaggl/60k-responses-of-16-personalities-test-mbt/data"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-[#8B1538] hover:text-[#C41E3A] underline break-all font-medium"
                            >
                                https://www.kaggle.com/datasets/anshulmehtakaggl/60k-responses-of-16-personalities-test-mbt/data
                            </a>
                        </div>

                        <div className="h-px bg-[#E8E5DC]" />

                        <div>
                            <h2 className="text-xl md:text-2xl font-bold text-[#4A7C7C] mb-3">
                                MBTI Descriptions
                            </h2>
                            <p className="text-gray-600 mb-2">
                                Source for personality characteristics and details:
                            </p>
                            <a
                                href="https://www.bsu.edu/about/administrativeoffices/careercenter/tools-resources/personality-types/"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-[#8B1538] hover:text-[#C41E3A] underline break-all font-medium"
                            >
                                https://www.bsu.edu/about/administrativeoffices/careercenter/tools-resources/personality-types/
                            </a>
                        </div>

                        <div className="h-px bg-[#E8E5DC]" />

                        <div>
                            <h2 className="text-xl md:text-2xl font-bold text-[#4A7C7C] mb-3">
                                Project Repository
                            </h2>
                            <p className="text-gray-600 mb-2">
                                Source code and documentation:
                            </p>
                            <a
                                href="https://github.com/DaRainFlavor/Multi-Class-Classification-of-Personality-Types-Using-Random-Forest"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-[#8B1538] hover:text-[#C41E3A] underline break-all font-medium"
                            >
                                https://github.com/DaRainFlavor/Multi-Class-Classification-of-Personality-Types-Using-Random-Forest
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}
