import React, { useState, useRef, useEffect } from "react";
import "./App.css";

// ===================================================================================
// CONFIGURATION
// Use the FastAPI server URL (running on port 8000 by default)
// ===================================================================================
const API_BASE = import.meta.env.VITE_API_BASE_URL ? `${import.meta.env.VITE_API_BASE_URL}/api` : "http://localhost:8000/api";
const HEALTH_CHECK_URL = import.meta.env.VITE_API_BASE_URL ? `${import.meta.env.VITE_API_BASE_URL}/health` : "http://localhost:8000/health";

// ===================================================================================
// Landing Page Component (Futuristic Update)
// ===================================================================================
const LandingPage = ({ onEnterChat }) => {
  return (
    <div className="min-h-screen farmer-gradient text-white flex flex-col items-center py-4 md:py-6 p-4 relative overflow-x-hidden selection:bg-emerald-500/30">
      {/* Animated Background Elements with Parallax-like feel */}
      <div className="fixed inset-0 bg-grid-pattern opacity-20 pointer-events-none transform translate-z-0"></div>
      <div className="fixed top-[-10%] left-[-10%] w-[50%] h-[50%] bg-emerald-500/10 rounded-full blur-[120px] animate-pulse pointer-events-none"></div>
      <div className="fixed bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-emerald-600/10 rounded-full blur-[120px] animate-pulse pointer-events-none"></div>

      {/* Scanning Line */}
      <div className="fixed left-0 right-0 h-[2px] bg-emerald-500/20 blur-sm animate-scan pointer-events-none z-0"></div>

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-emerald-500/30 rounded-full animate-float"
            style={{
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 5}s`,
              animationDuration: `${7 + Math.random() * 10}s`,
            }}
          ></div>
        ))}
      </div>

      <div className="text-center max-w-5xl mx-auto z-10 flex flex-col items-center min-h-[40vh] justify-center">
        <div className="mb-2 md:mb-3 animate-float">
          <div className="relative inline-block animate-pulse-glow rounded-full">
            <div className="absolute inset-0 bg-emerald-500/20 rounded-full blur-2xl animate-pulse"></div>
            <span className="text-4xl md:text-6xl filter drop-shadow-[0_0_30px_rgba(16,185,129,0.6)] relative z-10">
              🚜
            </span>
            <div className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-400 rounded-full animate-ping"></div>
          </div>
        </div>

        <h1
          className="text-4xl md:text-7xl font-black mb-2 tracking-tighter animate-slideUp"
          style={{ animationDelay: "0.1s" }}
        >
          <span className="bg-linear-to-r from-emerald-400 via-emerald-500 to-green-600 bg-clip-text text-transparent animate-gradient">
            KISSANSEVAAI
          </span>
        </h1>

        <p
          className="text-base md:text-xl mb-6 font-light text-slate-400 max-w-3xl mx-auto leading-relaxed px-4 animate-slideUp"
          style={{ animationDelay: "0.2s" }}
        >
          Your Digital Partner for{" "}
          <span className="text-emerald-400 font-medium">Smarter Farming</span>{" "}
          and Bumper Harvests.
        </p>

        {/* Scroll Indicator */}
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 animate-bounce opacity-40">
          <span className="text-[7px] uppercase tracking-[0.2em] text-emerald-500 font-bold animate-pulse">
            Explore
          </span>
          <div className="w-px h-4 bg-linear-to-b from-emerald-500 to-transparent"></div>
        </div>
      </div>

      <div className="w-full max-w-6xl mx-auto z-10 mt-2">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 md:gap-4 mb-8 w-full px-4">
          {/* Feature Cards */}
          {[
            {
              icon: "🌾",
              title: "Crop Intelligence",
              desc: "Expert guidance on crop selection and modern farming techniques.",
              delay: "0.3s",
            },
            {
              icon: "🔍",
              title: "Instant Diagnosis",
              desc: "Identify pests and diseases instantly with just a photo.",
              delay: "0.4s",
            },
            {
              icon: "🗣️",
              title: "Voice Support",
              desc: "Talk to your assistant naturally for hands-free farming help.",
              delay: "0.5s",
            },
          ].map((feature, i) => (
            <div
              key={i}
              className="futuristic-glass p-3 md:p-4 rounded-[1.2rem] futuristic-border transform hover:scale-105 transition-all duration-500 group animate-slideUp"
              style={{ animationDelay: feature.delay }}
            >
              <div className="text-2xl md:text-3xl mb-1 group-hover:scale-110 transition-transform group-hover:rotate-6">
                {feature.icon}
              </div>
              <h3
                className={`text-base md:text-lg font-bold mb-1 text-emerald-400 group-hover:text-emerald-300 transition-colors`}
              >
                {feature.title}
              </h3>
              <p className="text-slate-400 text-[10px] md:text-xs leading-relaxed group-hover:text-slate-300 transition-colors">
                {feature.desc}
              </p>
              {/* Card Glow Effect */}
              <div className="absolute inset-0 bg-emerald-500/5 opacity-0 group-hover:opacity-100 transition-opacity rounded-[1.2rem] pointer-events-none"></div>
            </div>
          ))}
        </div>

        <div className="flex justify-center mb-4">
          <button
            onClick={onEnterChat}
            className="group relative px-6 py-2.5 md:px-8 md:py-3 bg-transparent overflow-hidden rounded-full transition-all duration-500 animate-slideUp shadow-[0_0_20px_rgba(16,185,129,0.2)] hover:shadow-[0_0_40px_rgba(16,185,129,0.6)] animate-shimmer hover:scale-105 active:scale-95"
            style={{ animationDelay: "0.6s" }}
          >
            <div className="absolute inset-0 bg-linear-to-r from-emerald-500 via-green-600 to-emerald-700 transition-all duration-500 group-hover:scale-110"></div>
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 bg-[radial-gradient(circle_at_center,var(--tw-gradient-from)_0%,transparent_70%)] from-white/30 transition-opacity duration-300"></div>
            <div className="absolute inset-0 bg-grid-pattern opacity-10 group-hover:opacity-20 transition-opacity"></div>

            {/* Inner Border Glow */}
            <div className="absolute inset-[2px] rounded-full border border-white/10 pointer-events-none"></div>

            <span className="relative text-xs md:text-sm font-black tracking-[0.2em] flex items-center gap-2 text-white drop-shadow-lg">
              ENTER COMMAND CENTER{" "}
              <span className="group-hover:translate-x-1 transition-transform duration-500 text-emerald-300">
                →
              </span>
            </span>
          </button>
        </div>
      </div>

      <footer className="w-full py-4 md:py-6 text-center text-slate-500 text-[10px] md:text-xs tracking-[0.3em] uppercase mt-auto relative overflow-hidden border-t border-slate-800/20">
        <div className="absolute inset-0 flex items-center justify-center opacity-5 pointer-events-none">
          <div className="w-full h-[1px] bg-emerald-500 animate-scan"></div>
        </div>
        <span className="relative z-10">
          Empowering the Hands that Feed Us // KISSANSEVAAI v3.0.4 // SYSTEM
          STATUS:{" "}
          <span className="text-emerald-500/60 animate-pulse">OPTIMAL</span>
        </span>
      </footer>
    </div>
  );
};

// ===================================================================================
// Farmer Chatbot Component
// ===================================================================================

const FarmerChatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      content:
        "Welcome to your smart farming assistant! 🌱\n\nI'm here to help you with:\n• Smart crop recommendations\n• Disease & pest identification\n• Soil analysis and improvement\n• Weather-based farming advice\n• Harvest optimization\n• Organic farming solutions\n\nChoose how you'd like to interact with me using the tabs below!",
      isUser: false,
      type: "welcome",
      timestamp: new Date(),
    },
  ]);
  const [activeTab, setActiveTab] = useState("text");
  const [textInput, setTextInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("checking");
  const [showScrollButton, setShowScrollButton] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const imageInputRef = useRef(null);
  const voiceInputRef = useRef(null);
  const textInputRef = useRef(null);

  // --- State for basic context ---
  const [context, setContext] = useState({
    crop: "rice",
    location: "Kerala",
    season: "kharif",
  });

  const [predictionFeatures, setPredictionFeatures] = useState({
    N: 90,
    P: 42,
    K: 43,
    temperature: 20.8,
    humidity: 82.0,
    ph: 6.5,
    rainfall: 202.9,
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // useEffect(() => {
  //   scrollToBottom();
  // }, [messages]);

  const handleScroll = (e) => {
    const { scrollTop, scrollHeight, clientHeight } = e.target;
    const isAtBottom = scrollHeight - scrollTop <= clientHeight + 100;
    setShowScrollButton(!isAtBottom);
  };

  // --- END State Update ---

  const farmingCategories = {
    crops: [
      "what crops should i plant in summer",
      "crop rotation benefits",
      "greenhouse management",
    ],
    pests: [
      "how to prevent pest attacks",
      "how to control aphids",
      "organic farming tips",
    ],
    soil: [
      "soil preparation tips",
      "best fertilizer for vegetables",
      "composting",
    ],
    harvest: ["when to harvest tomatoes", "harvest timing", "seasonal care"],
    watering: [
      "watering schedule for plants",
      "irrigation tips",
      "water management",
    ],
  };

  const quickActions = [
    {
      icon: "🌱",
      text: "Plant Selection",
      query: "what crops should i plant in summer",
    },
    { icon: "🐛", text: "Pest Control", query: "how to prevent pest attacks" },
    { icon: "🌾", text: "Harvest Time", query: "when to harvest tomatoes" },
    {
      icon: "💧",
      text: "Watering Guide",
      query: "watering schedule for plants",
    },
  ];

  useEffect(() => {
    checkConnection();
  }, []);

  useEffect(() => {
    if (activeTab === "text" && textInputRef.current) {
      setTimeout(() => textInputRef.current.focus(), 100);
    }
  }, [activeTab]);

  const checkConnection = async () => {
    try {
      const response = await fetch(HEALTH_CHECK_URL);
      setConnectionStatus(response.ok ? "connected" : "error");
    } catch (error) {
      setConnectionStatus("error");
      addMessage(
        "🔧 Backend server not responding. Please start the Python FastAPI server on port 8000.",
        false,
        "system"
      );
    }
  };

  const addMessage = (content, isUser, type = "text") => {
    const newMessage = {
      id: Date.now() + Math.random(),
      content,
      isUser,
      type,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);
  };

  // ===============================================================================
  // UPDATED: sendTextMessage (FastAPI /api/chat endpoint -> Calls Gemini API)
  // Logic simplified for a standard, non-prediction chat interaction.
  // ===============================================================================
  const sendTextMessage = async (messageText = null) => {
    const message = messageText || textInput.trim();
    if (!message) return;

    // Add user message to chat
    addMessage(message, true, "text");
    setTextInput("");
    setIsLoading(true);

    const requestContext = {
      crop: context.crop,
      location: context.location,
      season: context.season,
      features: predictionFeatures,
    };

    const requestBody = {
      query: message,
      context: requestContext,
    };

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      setIsLoading(false);

      if (response.ok) {
        // ASSUMPTION: FastAPI now returns a simple object with the Gemini-generated text.
        // It should have an 'answer' field and may optionally include other fields like 'confidence'.
        let answerContent =
          data.answer || "I'm sorry, I couldn't get a response from the AI.";

        // Optional: Include intent/confidence if backend still returns it (see backend changes)
        if (data.confidence && data.intent) {
          const confidenceDisplay = (data.confidence * 100).toFixed(1);
          answerContent += `\n\n---`;
          answerContent += `\nConfidence: ${confidenceDisplay}% (Intent: ${data.intent})`;
          if (data.escalation_id) {
            answerContent += `\n⚠️ Low Confidence: This query was flagged for manual review (Escalation ID: ${data.escalation_id})`;
          }
        }

        addMessage(answerContent, false, "text");
      } else {
        // Handle HTTP error (e.g., 429 Rate Limit, 500 Server Error)
        const errorMessage = data.detail || "Unknown server error.";
        addMessage(`❌ Backend Error: ${errorMessage}`, false, "error");
      }
    } catch (error) {
      setIsLoading(false);
      addMessage(
        `🚨 Connection failed: ${error.message}. Is FastAPI server running?`,
        false,
        "error"
      );
    }
  };

  // ===============================================================================
  // uploadImage (No functional change)
  // ===============================================================================
  const uploadImage = async () => {
    const file = imageInputRef.current?.files[0];
    if (!file) {
      addMessage("📷 Please select an image first!", false, "system");
      return;
    }

    // Add user message to chat
    addMessage(
      `📸 Analyzing image: ${file.name} (Crop: ${context.crop})`,
      true,
      "image"
    );
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("crop", context.crop);
    formData.append("location", context.location);
    formData.append("season", context.season);

    try {
      const response = await fetch(`${API_BASE}/image`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setIsLoading(false);

      if (response.ok) {
        // Response matches ImageResponse model
        const confidenceDisplay = (data.confidence * 100).toFixed(1);
        let answerContent = `Analysis Result:\n\n`;
        answerContent += `Classification: ${data.label}\n`;
        answerContent += `Confidence: ${confidenceDisplay}%\n`;
        answerContent += `Remedy: ${data.remedy}`; // Added remedy from backend model
        answerContent += `\nModel: ${data.used_model}`;

        if (data.escalation_id) {
          answerContent += `\n\n⚠️ Low Confidence: This image analysis was flagged for manual review (Escalation ID: ${data.escalation_id})`;
        }

        addMessage(answerContent, false, "image");
      } else {
        const errorMessage =
          data.detail || "Unknown server error during image analysis.";
        addMessage(`❌ Backend Error: ${errorMessage}`, false, "error");
      }
    } catch (error) {
      setIsLoading(false);
      addMessage(`🚨 Upload failed: ${error.message}`, false, "error");
    }
    imageInputRef.current.value = "";
  };

  // ===============================================================================
  // uploadVoice (Calls FastAPI /api/voice)
  // ===============================================================================
  const uploadVoice = async () => {
    const file = voiceInputRef.current?.files[0];
    if (!file) {
      addMessage("🎤 Please select an audio file first!", false, "system");
      return;
    }

    addMessage(`🎤 Processing audio: ${file.name}`, true, "voice");
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("crop", context.crop);
    formData.append("location", context.location);
    formData.append("season", context.season);

    try {
      const response = await fetch(`${API_BASE}/voice`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setIsLoading(false);

      if (response.ok) {
        addMessage(data.answer, false, "voice");
      } else {
        const errorMessage =
          data.detail || "Unknown server error during voice processing.";
        addMessage(`❌ Backend Error: ${errorMessage}`, false, "error");
      }
    } catch (error) {
      setIsLoading(false);
      addMessage(`🚨 Upload failed: ${error.message}`, false, "error");
    }
    voiceInputRef.current.value = "";
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage();
    }
  };

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case "connected":
        return "🟢";
      case "error":
        return "🔴";
      default:
        return "🟡";
    }
  };

  const getMessageStyle = (message) => {
    if (message.isUser) {
      return "user-bubble text-white ml-auto rounded-2xl rounded-tr-none";
    }
    const styles = {
      welcome: "ai-bubble border-emerald-500/30",
      text: "ai-bubble border-slate-700/50",
      image: "ai-bubble border-purple-500/30",
      voice: "ai-bubble border-amber-500/30",
      system: "bg-slate-900/50 border-slate-800 text-slate-400",
      error: "bg-red-900/20 border-red-500/30 text-red-200",
    };
    return `${styles[message.type] || styles.text} rounded-2xl rounded-tl-none`;
  };

  const TabButton = ({ id, icon, label, isActive, onClick }) => (
    <button
      onClick={onClick}
      className={`flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl font-bold transition-all duration-300 ${
        isActive
          ? "bg-emerald-500/20 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.15)] border border-emerald-500/40"
          : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/30"
      }`}
    >
      <span className="text-xl">{icon}</span>
      <span className="hidden xl:block uppercase tracking-widest text-[10px]">
        {label}
      </span>
    </button>
  );

  return (
    <div className="h-screen farmer-gradient text-slate-200 font-sans selection:bg-emerald-500/30 overflow-hidden">
      <div className="max-w-[1800px] mx-auto h-full flex flex-col p-4 md:p-6 gap-6">
        {/* Futuristic Header */}
        <header className="futuristic-glass rounded-3xl p-4 md:p-6 futuristic-border flex items-center justify-between animate-fadeIn bg-slate-950/40 backdrop-blur-2xl">
          <div className="flex items-center gap-6">
            <div className="relative">
              <span className="text-5xl filter drop-shadow-[0_0_15px_rgba(16,185,129,0.5)]">
                🚜
              </span>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 rounded-full border-2 border-[#020617]"></div>
            </div>
            <div className="hidden sm:block">
              <h1 className="text-2xl md:text-3xl font-black tracking-tighter text-white">
                KISSANSEVAAI
              </h1>
              <div className="flex items-center gap-3 text-[10px] uppercase tracking-[0.3em] text-emerald-500/60 font-bold">
                <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_#10b981]"></span>
                <span>Precision Agriculture Intelligence</span>
                <span className="text-slate-700">|</span>
                <span className="text-emerald-400/40 animate-pulse">
                  Live Data Stream Active
                </span>
              </div>
            </div>
          </div>

          {/* Navigation Tabs - Shifted to Navbar */}
          <nav className="flex items-center gap-2 bg-slate-900/40 p-1.5 rounded-2xl border border-slate-800/50">
            <TabButton
              id="text"
              icon="💬"
              label="AI Advisor"
              isActive={activeTab === "text"}
              onClick={() => setActiveTab("text")}
            />
            <TabButton
              id="image"
              icon="📸"
              label="Vision Scan"
              isActive={activeTab === "image"}
              onClick={() => setActiveTab("image")}
            />
            <TabButton
              id="voice"
              icon="🎤"
              label="Audio Link"
              isActive={activeTab === "voice"}
              onClick={() => setActiveTab("voice")}
            />
            <TabButton
              id="predict"
              icon="📊"
              label="Yield Predict"
              isActive={activeTab === "predict"}
              onClick={() => setActiveTab("predict")}
            />
          </nav>

          {/* Context Monitor - Real-time Environmental Data */}
          <div className="hidden xl:flex items-center gap-8 px-8 border-x border-slate-800/30">
            {[
              { label: "Crop", value: context.crop, color: "text-emerald-400" },
              {
                label: "Location",
                value: context.location,
                color: "text-blue-400",
              },
              {
                label: "Season",
                value: context.season,
                color: "text-purple-400",
              },
              {
                label: "Soil (N-P-K)",
                value: `${predictionFeatures.N}-${predictionFeatures.P}-${predictionFeatures.K}`,
                color: "text-emerald-400",
              },
            ].map((item, i) => (
              <div key={i} className="flex flex-col">
                <span className="text-[8px] uppercase tracking-[0.3em] text-slate-500 mb-0.5">
                  {item.label}
                </span>
                <span
                  className={`text-[10px] font-black uppercase tracking-widest ${item.color} animate-pulse`}
                >
                  {item.value || "---"}
                </span>
              </div>
            ))}
          </div>

          <div className="hidden lg:flex items-center gap-8">
            <div className="text-right">
              <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">
                System Status
              </div>
              <div
                className={`text-sm font-bold flex items-center gap-2 justify-end ${
                  connectionStatus === "connected"
                    ? "text-emerald-400"
                    : "text-red-400"
                }`}
              >
                {connectionStatus === "connected"
                  ? "SYNCHRONIZED"
                  : "DISCONNECTED"}
                <div
                  className={`w-2 h-2 rounded-full ${
                    connectionStatus === "connected"
                      ? "bg-emerald-500 shadow-[0_0_10px_#10b981]"
                      : "bg-red-500"
                  }`}
                ></div>
              </div>
            </div>
          </div>
        </header>

        <main className="flex-1 flex flex-col lg:flex-row gap-6 min-h-0">
          {/* Left Sidebar - Controls & Context - Shrunk */}
          <aside
            className="lg:w-64 flex flex-col gap-6 animate-fadeIn"
            style={{ animationDelay: "0.1s" }}
          >
            {/* Context Panel */}
            <section className="futuristic-glass rounded-3xl futuristic-border flex-1 flex-col overflow-hidden hidden lg:flex">
              <div className="p-6 border-b border-slate-800/50">
                <h3 className="text-xs uppercase tracking-[0.2em] text-emerald-500 font-bold flex items-center gap-2">
                  <span className="w-4 h-px bg-emerald-500"></span>{" "}
                  Environmental Data
                </h3>
              </div>

              <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
                {[
                  {
                    label: "Target Crop",
                    value: context.crop,
                    key: "crop",
                    icon: "🌱",
                  },
                  {
                    label: "Geospatial Location",
                    value: context.location,
                    key: "location",
                    icon: "📍",
                  },
                  {
                    label: "Temporal Season",
                    value: context.season,
                    key: "season",
                    icon: "⏳",
                  },
                ].map((item) => (
                  <div key={item.key} className="group">
                    <label className="text-[10px] uppercase tracking-widest text-slate-500 mb-2 block group-focus-within:text-emerald-400 transition-colors">
                      {item.label}
                    </label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">
                        {item.icon}
                      </span>
                      <input
                        type="text"
                        value={item.value}
                        onChange={(e) =>
                          setContext((prev) => ({
                            ...prev,
                            [item.key]: e.target.value,
                          }))
                        }
                        className="w-full bg-slate-900/50 border border-slate-800 rounded-xl py-3 pl-10 pr-4 text-sm focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 transition-all"
                      />
                    </div>
                  </div>
                ))}

                {/* Soil Nutrients & Environment */}
                <div className="pt-4 border-t border-slate-800/50 space-y-4">
                  <div className="grid grid-cols-3 gap-2">
                    {["N", "P", "K"].map((nutrient) => (
                      <div key={nutrient}>
                        <label className="text-[8px] uppercase tracking-widest text-slate-600 mb-1 block">
                          {nutrient}
                        </label>
                        <input
                          type="number"
                          value={predictionFeatures[nutrient]}
                          onChange={(e) =>
                            setPredictionFeatures((prev) => ({
                              ...prev,
                              [nutrient]: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full bg-slate-900/30 border border-slate-800 rounded-lg py-1.5 px-2 text-[10px] focus:border-emerald-500/50 outline-none"
                        />
                      </div>
                    ))}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: "Temp", key: "temperature", unit: "°C" },
                      { label: "Humid", key: "humidity", unit: "%" },
                      { label: "pH", key: "ph", unit: "" },
                      { label: "Rain", key: "rainfall", unit: "mm" },
                    ].map((feat) => (
                      <div key={feat.key}>
                        <label className="text-[8px] uppercase tracking-widest text-slate-600 mb-1 block">
                          {feat.label} {feat.unit}
                        </label>
                        <input
                          type="number"
                          value={predictionFeatures[feat.key]}
                          onChange={(e) =>
                            setPredictionFeatures((prev) => ({
                              ...prev,
                              [feat.key]: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full bg-slate-900/30 border border-slate-800 rounded-lg py-1.5 px-2 text-[10px] focus:border-emerald-500/50 outline-none"
                        />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Knowledge Base - Moved to Left Sidebar */}
                <div className="pt-6 border-t border-slate-800/50">
                  <h3 className="text-xs uppercase tracking-[0.2em] text-emerald-500 font-bold mb-4 flex items-center gap-2">
                    <span className="w-4 h-px bg-emerald-500"></span> Knowledge
                  </h3>
                  <div className="space-y-2">
                    {Object.keys(farmingCategories)
                      .slice(0, 4)
                      .map((cat) => (
                        <button
                          key={cat}
                          onClick={() =>
                            sendTextMessage(
                              `Provide detailed ${cat} protocols and best practices for my current context.`
                            )
                          }
                          className="w-full p-3 bg-slate-900/30 border border-slate-800/50 rounded-xl text-[10px] uppercase tracking-widest text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-all text-left group"
                        >
                          <span className="group-hover:text-emerald-400 transition-colors">
                            {cat} Protocol
                          </span>
                        </button>
                      ))}
                  </div>
                </div>
              </div>

              <div className="p-6 border-t border-slate-800/50 bg-slate-900/20">
                <div className="text-[10px] uppercase tracking-widest text-emerald-500 mb-2">
                  AI Core Load
                </div>
                <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 w-[65%] animate-pulse"></div>
                </div>
              </div>
            </section>
          </aside>

          {/* Center - Chat Interface */}
          <section
            className="flex-1 futuristic-glass rounded-3xl futuristic-border flex flex-col min-h-0 animate-slideUp relative overflow-hidden"
            style={{ animationDelay: "0.2s" }}
          >
            {/* Initial Scan Effect on Load */}
            <div className="absolute inset-0 pointer-events-none z-50 opacity-20">
              <div className="w-full h-[2px] bg-emerald-500 animate-scan"></div>
            </div>

            {/* Chat Messages Area */}
            <div
              ref={chatContainerRef}
              onScroll={handleScroll}
              className="flex-1 overflow-y-auto p-6 space-y-6 chat-container relative"
            >
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.isUser ? "justify-end" : "justify-start"
                  } animate-message`}
                >
                  <div
                    className={`max-w-[85%] p-5 rounded-3xl shadow-2xl transition-all duration-300 ${getMessageStyle(
                      message
                    )}`}
                  >
                    <div className="flex items-center gap-3 mb-3 opacity-60">
                      <span className="text-xl">
                        {message.isUser ? "👤" : "🚜"}
                      </span>
                      <span className="text-[10px] uppercase tracking-[0.2em] font-bold">
                        {message.isUser
                          ? "Authorized Farmer"
                          : "KissanSeva Intelligence"}
                      </span>
                      <span className="text-[10px] ml-auto">
                        {message.timestamp.toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                    </div>
                    <div className="text-sm leading-relaxed whitespace-pre-line font-medium">
                      {message.content}
                    </div>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start animate-message">
                  <div className="futuristic-glass p-5 rounded-3xl futuristic-border border-emerald-500/30 overflow-hidden relative">
                    <div className="absolute top-0 left-0 w-full h-1 bg-emerald-500/10">
                      <div className="h-full bg-emerald-500 w-1/3 animate-loading-bar"></div>
                    </div>
                    <div className="flex items-center gap-4 text-emerald-400">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                      </div>
                      <span className="text-[10px] uppercase tracking-widest font-bold">
                        AI is analyzing your request...
                      </span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />

              {/* Scroll to Bottom Button */}
              {showScrollButton && (
                <button
                  onClick={scrollToBottom}
                  className="fixed bottom-32 right-12 bg-emerald-500/20 hover:bg-emerald-500/40 text-emerald-400 p-4 rounded-full backdrop-blur-xl border border-emerald-500/30 shadow-[0_0_20px_rgba(16,185,129,0.2)] transition-all animate-bounce z-50 group"
                >
                  <span className="text-xl group-hover:translate-y-1 transition-transform block">
                    ↓
                  </span>
                </button>
              )}
            </div>

            {/* Input Area */}
            <footer className="p-6 bg-slate-900/30 border-t border-slate-800/50">
              {activeTab === "text" && (
                <div className="relative group">
                  <textarea
                    ref={textInputRef}
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Enter command or ask a question..."
                    rows="1"
                    className="w-full bg-slate-900/80 border-2 border-slate-800 rounded-2xl py-5 pl-6 pr-32 text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500/50 focus:ring-4 focus:ring-emerald-500/10 transition-all resize-none"
                  />
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 flex gap-2">
                    <button
                      onClick={() => setTextInput("")}
                      className="p-3 text-slate-500 hover:text-slate-300 transition-colors"
                    >
                      <span className="text-xs uppercase tracking-widest font-bold">
                        Reset
                      </span>
                    </button>
                    <button
                      onClick={() => sendTextMessage()}
                      disabled={isLoading || !textInput.trim()}
                      className="bg-linear-to-r from-emerald-600 to-green-700 hover:from-emerald-500 hover:to-green-600 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-600 text-white px-6 py-3 rounded-xl font-bold text-xs uppercase tracking-widest transition-all duration-300 shadow-lg shadow-emerald-500/20 active:scale-95"
                    >
                      {isLoading ? "..." : "Execute"}
                    </button>
                  </div>
                </div>
              )}

              {activeTab === "image" && (
                <div className="flex gap-4 animate-fadeIn">
                  <button
                    onClick={() => imageInputRef.current?.click()}
                    className="flex-1 bg-slate-900/80 border-2 border-dashed border-slate-700 rounded-2xl p-4 flex items-center justify-center gap-4 hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-all group"
                  >
                    <input
                      ref={imageInputRef}
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => {
                        if (e.target.files[0])
                          document.getElementById(
                            "img-status"
                          ).innerText = `READY: ${e.target.files[0].name}`;
                      }}
                    />
                    <span className="text-3xl group-hover:scale-110 transition-transform">
                      📸
                    </span>
                    <div className="text-left">
                      <div className="text-xs font-bold uppercase tracking-widest text-slate-400">
                        Initialize Vision Scan
                      </div>
                      <div
                        id="img-status"
                        className="text-[10px] text-slate-600"
                      >
                        No file selected
                      </div>
                    </div>
                  </button>
                  <button
                    onClick={uploadImage}
                    disabled={isLoading}
                    className="bg-linear-to-r from-emerald-600 to-green-700 px-10 rounded-2xl font-bold text-xs uppercase tracking-widest transition-all hover:scale-105 active:scale-95 shadow-lg shadow-emerald-500/20"
                  >
                    Analyze
                  </button>
                </div>
              )}

              {activeTab === "voice" && (
                <div className="flex gap-4 animate-fadeIn">
                  <button
                    onClick={() => voiceInputRef.current?.click()}
                    className="flex-1 bg-slate-900/80 border-2 border-dashed border-slate-700 rounded-2xl p-4 flex items-center justify-center gap-4 hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-all group"
                  >
                    <input
                      ref={voiceInputRef}
                      type="file"
                      accept="audio/*"
                      className="hidden"
                      onChange={(e) => {
                        if (e.target.files[0])
                          document.getElementById(
                            "voice-status"
                          ).innerText = `READY: ${e.target.files[0].name}`;
                      }}
                    />
                    <span className="text-3xl group-hover:scale-110 transition-transform">
                      🎤
                    </span>
                    <div className="text-left">
                      <div className="text-xs font-bold uppercase tracking-widest text-slate-400">
                        Initialize Audio Link
                      </div>
                      <div
                        id="voice-status"
                        className="text-[10px] text-slate-600"
                      >
                        No file selected
                      </div>
                    </div>
                  </button>
                  <button
                    onClick={uploadVoice}
                    disabled={isLoading}
                    className="bg-linear-to-r from-emerald-600 to-green-700 px-10 rounded-2xl font-bold text-xs uppercase tracking-widest transition-all hover:scale-105 active:scale-95 shadow-lg shadow-emerald-500/20"
                  >
                    Process
                  </button>
                </div>
              )}

              {activeTab === "predict" && (
                <div className="flex gap-4 animate-fadeIn">
                  <div className="flex-1 bg-slate-900/80 border border-slate-700 rounded-2xl p-4 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-4">
                      <span className="text-3xl">📊</span>
                      <div className="text-left">
                        <div className="text-xs font-bold uppercase tracking-widest text-slate-400">
                          Yield Prediction Engine
                        </div>
                        <div className="text-[10px] text-slate-600">
                          Ready to analyze environmental features
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      {["N", "P", "K"].map((n) => (
                        <div
                          key={n}
                          className="px-2 py-1 bg-slate-800 rounded text-[8px] font-bold text-emerald-500"
                        >
                          {n}: {predictionFeatures[n]}
                        </div>
                      ))}
                    </div>
                  </div>
                  <button
                    onClick={() =>
                      sendTextMessage(
                        "Based on my current environmental data (N, P, K, Temp, Humidity, pH, Rainfall), please provide a detailed yield prediction and optimization strategy."
                      )
                    }
                    disabled={isLoading}
                    className="bg-linear-to-r from-emerald-600 to-green-700 px-10 rounded-2xl font-bold text-xs uppercase tracking-widest transition-all hover:scale-105 active:scale-95 shadow-lg shadow-emerald-500/20"
                  >
                    Predict
                  </button>
                </div>
              )}
            </footer>
          </section>

          {/* Right - Quick Action Icons (Outside Chat) */}
          <div
            className="hidden xl:flex flex-col gap-4 justify-center animate-fadeIn"
            style={{ animationDelay: "0.3s" }}
          >
            {quickActions.map((action, index) => (
              <button
                key={index}
                onClick={() => sendTextMessage(action.query)}
                title={action.text}
                className="w-12 h-12 flex items-center justify-center bg-slate-900/50 border border-slate-800 rounded-2xl hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-all group shadow-lg"
              >
                <div className="text-2xl group-hover:scale-110 transition-transform">
                  {action.icon}
                </div>
              </button>
            ))}
          </div>
        </main>
      </div>
      <style jsx>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        .animate-slideIn {
          animation: slideIn 0.3s ease-out;
        }
        .scale-102:hover {
          transform: scale(1.02);
        }
        .border-3 {
          border-width: 3px;
        }
      `}</style>
    </div>
  );
};

// ===================================================================================
// Main App Component
// ===================================================================================
const App = () => {
  const [showChat, setShowChat] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);

  const handleEnterChat = () => {
    setIsInitializing(true);
    setTimeout(() => {
      setShowChat(true);
      setIsInitializing(false);
    }, 1500);
  };

  if (isInitializing) {
    return (
      <div className="h-screen farmer-gradient flex flex-col items-center justify-center text-white relative overflow-hidden">
        <div className="fixed inset-0 bg-grid-pattern opacity-20"></div>
        <div className="relative z-10 flex flex-col items-center">
          <div className="text-8xl mb-8 animate-bounce filter drop-shadow-[0_0_30px_rgba(16,185,129,0.6)]">
            🚜
          </div>
          <h2 className="text-2xl font-black tracking-[0.5em] uppercase mb-4 animate-pulse">
            Initializing AI Core
          </h2>
          <div className="w-64 h-1 bg-slate-800 rounded-full overflow-hidden relative">
            <div className="absolute inset-0 bg-emerald-500/20"></div>
            <div className="h-full bg-emerald-500 w-full animate-loading-bar"></div>
          </div>
          <div className="mt-6 text-[10px] uppercase tracking-[0.3em] text-emerald-500/60 font-bold">
            Establishing Secure Satellite Link...
          </div>
        </div>
        <div className="fixed left-0 right-0 h-[2px] bg-emerald-500/20 blur-sm animate-scan"></div>
      </div>
    );
  }

  if (showChat) {
    return <FarmerChatbot />;
  }

  return <LandingPage onEnterChat={handleEnterChat} />;
};

// Make sure to export the main App component
export default App;
