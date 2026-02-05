"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { questions, answerOptions } from "@/data/questions";

const getPositionFromValue = (value: number | null) => {
    if (value === null) return null;
    return value + 3;
};

const getValueFromPosition = (position: number) => {
    return position - 3;
};

export default function QuizPage() {
    const router = useRouter();
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [answers, setAnswers] = useState<(number | null)[]>(
        new Array(questions.length).fill(null)
    );
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const sliderRef = useRef<HTMLDivElement>(null);

    const progress = ((currentQuestion + 1) / questions.length) * 100;
    const currentAnswer = answers[currentQuestion];

    const handleAnswer = useCallback((value: number) => {
        setAnswers((prev) => {
            const newAnswers = [...prev];
            newAnswers[currentQuestion] = value;
            return newAnswers;
        });
    }, [currentQuestion]);

    const handleDrag = useCallback((clientX: number) => {
        if (!sliderRef.current) return;

        const rect = sliderRef.current.getBoundingClientRect();
        const x = clientX - rect.left;
        const percentage = Math.max(0, Math.min(1, x / rect.width));

        const position = Math.round(percentage * 6);
        const value = getValueFromPosition(position);
        handleAnswer(value);
    }, [handleAnswer]);

    const handleMouseDown = (e: React.MouseEvent) => {
        setIsDragging(true);
        handleDrag(e.clientX);
    };

    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (isDragging) {
            handleDrag(e.clientX);
        }
    }, [isDragging, handleDrag]);

    const handleMouseUp = useCallback(() => {
        setIsDragging(false);
    }, []);

    const handleTouchStart = (e: React.TouchEvent) => {
        setIsDragging(true);
        handleDrag(e.touches[0].clientX);
    };

    const handleTouchMove = useCallback((e: TouchEvent) => {
        if (isDragging) {
            handleDrag(e.touches[0].clientX);
        }
    }, [isDragging, handleDrag]);

    useEffect(() => {
        if (isDragging) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
            window.addEventListener('touchmove', handleTouchMove);
            window.addEventListener('touchend', handleMouseUp);
        }
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
            window.removeEventListener('touchmove', handleTouchMove);
            window.removeEventListener('touchend', handleMouseUp);
        };
    }, [isDragging, handleMouseMove, handleMouseUp, handleTouchMove]);

    const goToNext = useCallback(() => {
        if (currentQuestion < questions.length - 1) {
            setCurrentQuestion((prev) => prev + 1);
        }
    }, [currentQuestion]);

    const goToPrevious = useCallback(() => {
        if (currentQuestion > 0) {
            setCurrentQuestion((prev) => prev - 1);
        }
    }, [currentQuestion]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.ctrlKey) {
                if (e.key === "ArrowLeft") {
                    goToPrevious();
                } else if (e.key === "ArrowRight") {
                    goToNext();
                }
            } else {
                const key = parseInt(e.key);
                if (!isNaN(key) && key >= 1 && key <= 7) {
                    handleAnswer(answerOptions[key - 1].value);
                }
            }
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [goToNext, goToPrevious, handleAnswer]);

    const handleSubmit = async () => {
        const unansweredCount = answers.filter((a) => a === null).length;
        if (unansweredCount > 0) {
            alert(`Please answer all questions. ${unansweredCount} questions remaining.`);
            return;
        }

        setIsSubmitting(true);

        try {
            const isLocalhost = typeof window !== 'undefined' &&
                (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
            const apiUrl = isLocalhost
                ? 'http://localhost:5000/predict'
                : '/api/predict';

            const response = await fetch(apiUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ answers }),
            });

            if (!response.ok) {
                const errData = (await response.json().catch(() => ({}))) as { error?: string };
                throw new Error(errData.error || "Failed to get prediction");
            }

            const data = await response.json();
            sessionStorage.setItem("quizResult", JSON.stringify(data));
            router.push("/results");
        } catch (error) {
            console.error("Error submitting quiz:", error);
            const errorMessage = error instanceof Error ? error.message : "Unknown error";
            alert(`Failed to submit quiz: ${errorMessage}`);
            setIsSubmitting(false);
        }
    };

    const jumpToQuestion = (index: number) => {
        setCurrentQuestion(index);
        setIsMenuOpen(false);
    };

    const answeredCount = answers.filter((a) => a !== null).length;
    const isLastQuestion = currentQuestion === questions.length - 1;
    const currentPosition = getPositionFromValue(currentAnswer);

    const isLongText = questions[currentQuestion].text.length > 80;

    return (
        <main className="h-screen overflow-hidden bg-[#F5F2E8] text-[#2C2C2C]">
            {/* Progress bar */}
            <div className="fixed top-0 left-0 right-0 h-2 bg-[#E8E5DC] z-50">
                <div
                    className="h-full bg-[#C4A52D] transition-all duration-300"
                    style={{ width: `${progress}%` }}
                />
            </div>

            {/* Header - Darker Background (Burgundy) */}
            <header className="fixed top-0 left-0 right-0 p-3 md:p-4 z-40 bg-[#8B1538] border-b-2 border-[#C4A52D] shadow-md">
                <div className="max-w-6xl mx-auto flex items-center justify-between">
                    <div className="flex items-center gap-2 md:gap-4">
                        {/* Hamburger Menu Button - Mobile Only - White outline since bg is burgundy */}
                        <button
                            onClick={() => setIsMenuOpen(!isMenuOpen)}
                            className="lg:hidden p-2 rounded-lg bg-white/10 text-white hover:bg-white/20 transition-all"
                            aria-label="Toggle questions menu"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                {isMenuOpen ? (
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                ) : (
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                                )}
                            </svg>
                        </button>

                        <button
                            onClick={() => router.push("/")}
                            className="text-white hover:text-[#C4A52D] font-medium transition-colors text-sm md:text-base flex items-center gap-1"
                        >
                            <span>←</span> <span className="hidden xs:inline">Back</span>
                        </button>
                        <span className="text-[#C4A52D] hidden md:inline">|</span>
                        <span className="text-xs md:text-sm text-[#E8E5DC] font-medium">
                            Q{currentQuestion + 1}/{questions.length}
                        </span>
                    </div>
                    <div className="text-xs md:text-sm text-[#C4A52D] font-bold flex flex-col items-end">
                        <span>{answeredCount}/{questions.length} Answered</span>
                        <span className="text-[10px] text-white/50 font-normal">v1.1</span>
                    </div>
                </div>
            </header>

            {/* Mobile Menu Overlay */}
            {isMenuOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-40 lg:hidden"
                    onClick={() => setIsMenuOpen(false)}
                />
            )}

            {/* Mobile Slide-out Menu */}
            <div className={`fixed top-0 left-0 h-full w-72 bg-[#F5F2E8] z-50 transform transition-transform duration-300 lg:hidden ${isMenuOpen ? 'translate-x-0' : '-translate-x-full'
                }`}>
                <div className="p-4 pt-4">
                    <div className="flex justify-between items-center mb-6">
                        <p className="text-lg text-[#8B1538] font-bold">Questions</p>
                        <button
                            onClick={() => setIsMenuOpen(false)}
                            className="p-2 rounded-lg bg-[#E8E5DC] text-[#4A7C7C]"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>

                    <div className="grid grid-cols-5 gap-2 mb-4 overflow-y-auto max-h-[70vh] custom-scrollbar">
                        {questions.map((_, i) => (
                            <button
                                key={i}
                                onClick={() => jumpToQuestion(i)}
                                className={`w-10 h-10 rounded-lg transition-all text-xs font-bold flex items-center justify-center ${i === currentQuestion
                                    ? "bg-[#8B1538] text-white scale-110 shadow-md"
                                    : answers[i] !== null
                                        ? "bg-[#4A7C7C] text-white"
                                        : "bg-[#E8E5DC] text-[#888] hover:bg-[#C4A52D] hover:text-white"
                                    }`}
                            >
                                {i + 1}
                            </button>
                        ))}
                    </div>

                    <div className="pt-4 border-t border-[#E8E5DC]">
                        <div className="flex justify-between text-xs text-[#4A7C7C]">
                            <span>Progress</span>
                            <span className="font-bold text-[#8B1538]">{answeredCount}/60</span>
                        </div>
                        <div className="mt-2 h-2 bg-[#E8E5DC] rounded-full overflow-hidden">
                            <div
                                className="h-full bg-[#4A7C7C] transition-all duration-300"
                                style={{ width: `${(answeredCount / 60) * 100}%` }}
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Main content with side panel */}
            <div className="pt-20 md:pt-24 pb-0 px-3 md:px-4 lg:ml-72 h-full">
                <div className="max-w-4xl mx-auto h-full flex flex-col">

                    {/* Left side - Question Navigator (Desktop Only - Fixed Sidebar) */}
                    <div className="hidden lg:block fixed left-0 top-0 bottom-0 w-72 bg-white border-r-2 border-[#E8E5DC] pt-24 pb-8 px-6 overflow-y-auto z-30 shadow-lg">
                        <div className="flex flex-col h-full">
                            <p className="text-lg text-[#8B1538] font-bold mb-6 flex items-center gap-2">
                                <span>Questions</span>
                                <span className="text-xs bg-[#E8E5DC] text-[#2C2C2C] px-2 py-1 rounded-full">{answeredCount}/60</span>
                            </p>

                            <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                                <div className="grid grid-cols-5 gap-2">
                                    {questions.map((_, i) => (
                                        <button
                                            key={i}
                                            onClick={() => setCurrentQuestion(i)}
                                            className={`w-10 h-10 rounded-lg transition-all text-xs font-bold flex items-center justify-center ${i === currentQuestion
                                                ? "bg-[#8B1538] text-white shadow-md ring-2 ring-[#C4A52D] ring-offset-1"
                                                : answers[i] !== null
                                                    ? "bg-[#4A7C7C] text-white"
                                                    : "bg-[#F5F2E8] text-[#888] hover:bg-[#C4A52D] hover:text-white"
                                                }`}
                                        >
                                            {i + 1}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="mt-6 pt-6 border-t border-[#E8E5DC]">
                                <div className="flex justify-between text-xs text-[#4A7C7C] mb-2">
                                    <span>Progress</span>
                                    <span className="font-bold">{Math.round((answeredCount / 60) * 100)}%</span>
                                </div>
                                <div className="h-2 bg-[#E8E5DC] rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-[#4A7C7C] transition-all duration-300"
                                        style={{ width: `${(answeredCount / 60) * 100}%` }}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Right side - Question and Slider */}
                    <div className="flex-1 w-full flex flex-col overflow-hidden pb-4">
                        {/* Question Container */}
                        <div className="bg-white rounded-2xl md:rounded-3xl p-4 md:p-8 border-2 border-[#8B1538] shadow-lg mb-4 flex-1 flex flex-col min-h-0">
                            <div className="flex items-center justify-between mb-4 shrink-0">
                                <div className="inline-block px-4 py-1.5 bg-[#C4A52D] text-white text-xs md:text-sm font-bold rounded-full">
                                    Question {currentQuestion + 1}
                                </div>
                                <span className="text-xs md:text-sm text-[#4A7C7C] font-medium hidden md:inline-block">Select the option that best fits you</span>
                            </div>
                            <div className="flex-1 flex items-center justify-center overflow-y-auto w-full custom-scrollbar">
                                <h2 className={`${isLongText ? "text-xl md:text-2xl" : "text-2xl md:text-3xl lg:text-4xl"} font-medium text-[#2C2C2C] leading-tight md:leading-relaxed text-center`}>
                                    {questions[currentQuestion].text}
                                </h2>
                            </div>
                        </div>

                        {/* Slider - No Container */}
                        <div className="px-2 md:px-8 py-2 md:py-4 relative shrink-0">
                            {/* Draggable Slider Track */}
                            <div
                                ref={sliderRef}
                                className="relative h-12 cursor-pointer select-none touch-none"
                                onMouseDown={handleMouseDown}
                                onTouchStart={handleTouchStart}
                            >
                                {/* Background line */}
                                <div className="absolute top-1/2 left-0 right-0 h-2 bg-[#E8E5DC] rounded-full -translate-y-1/2" />

                                {/* Filled/Highlighted line */}
                                {currentPosition !== null && (
                                    <div
                                        className="absolute top-1/2 left-0 h-2 bg-[#8B1538] rounded-full -translate-y-1/2 transition-all duration-150"
                                        style={{ width: `${(currentPosition / 6) * 100}%` }}
                                    />
                                )}

                                {/* Dots/Points */}
                                <div className="absolute top-1/2 left-0 right-0 -translate-y-1/2 flex justify-between">
                                    {answerOptions.map((option, index) => (
                                        <button
                                            key={option.value}
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleAnswer(option.value);
                                            }}
                                            className={`w-5 h-5 md:w-6 md:h-6 rounded-full border-3 md:border-4 transition-all duration-150 z-10 ${currentPosition === index
                                                ? "bg-[#8B1538] border-[#8B1538] scale-150 shadow-lg"
                                                : currentPosition !== null && index < currentPosition
                                                    ? "bg-[#8B1538] border-[#8B1538]"
                                                    : "bg-white border-[#C4A52D] hover:border-[#8B1538] hover:scale-110"
                                                }`}
                                        />
                                    ))}
                                </div>
                            </div>

                            {/* Labels below dots */}
                            <div className="flex justify-between mt-3 px-0 select-none">
                                {answerOptions.map((option, index) => (
                                    <button
                                        key={option.value}
                                        onClick={() => handleAnswer(option.value)}
                                        className={`text-center w-12 md:w-20 leading-tight transition-all h-10 md:h-14 flex items-start justify-center pt-1 ${currentPosition === index
                                            ? "text-[#8B1538] font-bold text-xs md:text-base scale-105"
                                            : "text-[#4A7C7C] font-medium text-[10px] md:text-xs hover:text-[#8B1538]"
                                            }`}
                                    >
                                        <span className="block">{option.label}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Navigation Buttons - Below Slider */}
                        <div className="flex items-center justify-between mt-4 px-2 md:px-4 shrink-0">
                            <button
                                onClick={goToPrevious}
                                disabled={currentQuestion === 0}
                                className={`flex items-center gap-2 px-5 md:px-8 py-3 md:py-4 rounded-xl font-bold text-sm md:text-lg transition-all shadow-sm ${currentQuestion === 0
                                    ? "bg-[#E8E5DC] text-[#A0A0A0] cursor-not-allowed"
                                    : "bg-[#4A7C7C] text-white hover:bg-[#2D5A5A] hover:shadow-md active:scale-95"
                                    }`}
                            >
                                <span>←</span> Previous
                            </button>

                            {isLastQuestion ? (
                                <button
                                    onClick={handleSubmit}
                                    disabled={isSubmitting}
                                    className={`px-8 md:px-12 py-3 md:py-4 rounded-xl font-bold text-sm md:text-lg transition-all shadow-md ${isSubmitting
                                        ? "bg-[#E8E5DC] text-[#A0A0A0] cursor-not-allowed"
                                        : "bg-[#C41E3A] text-white hover:bg-[#A01830] hover:shadow-lg active:scale-95"
                                        }`}
                                >
                                    {isSubmitting ? "Processing..." : "Finish Quiz"}
                                </button>
                            ) : (
                                <button
                                    onClick={goToNext}
                                    className="flex items-center gap-2 px-8 md:px-12 py-3 md:py-4 rounded-xl font-bold text-sm md:text-lg bg-[#8B1538] text-white hover:bg-[#6B1028] transition-all shadow-md hover:shadow-lg active:scale-95"
                                >
                                    Next <span>→</span>
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}
