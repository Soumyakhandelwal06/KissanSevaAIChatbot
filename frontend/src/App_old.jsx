import React, { useState, useRef, useEffect } from "react";

// ===================================================================================
// NEW: Landing Page Component
// This component serves as the welcome screen for your application.
// ===================================================================================
const LandingPage = ({ onEnterChat }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-300 via-sky-400 to-cyan-500 text-white flex flex-col items-center justify-center p-4 overflow-hidden">
      <div className="text-center max-w-4xl mx-auto animate-fadeIn">
        <div className="mb-4">
          <span className="text-8xl">🌾</span>
        </div>
        <h1
          className="text-5xl md:text-7xl font-extrabold mb-4 bg-gradient-to-r from-white to-cyan-100 bg-clip-text text-transparent"
        >
          Cultivate Smarter. Grow More.
        </h1>
        <p className="text-xl md:text-2xl mb-8 font-light text-blue-100 max-w-2xl mx-auto">
          Your personal AI agronomist for a bountiful harvest.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {/* Feature Cards */}
          <div className="bg-white/20 backdrop-blur-md p-6 rounded-2xl border border-white/30 transform hover:scale-105 transition-transform duration-300">
            <div className="text-4xl mb-3">🌱</div>
            <h3 className="text-xl font-semibold mb-2">Smart Crop Advice</h3>
            <p className="text-blue-200 text-sm">
              Get recommendations on what to plant based on your soil and local
              weather.
            </p>
          </div>
          <div className="bg-white/20 backdrop-blur-md p-6 rounded-2xl border border-white/30 transform hover:scale-105 transition-transform duration-300">
            <div className="text-4xl mb-3">📸</div>
            <h3 className="text-xl font-semibold mb-2">Disease Detection</h3>
            <p className="text-blue-200 text-sm">
              Upload a photo of your plant to instantly identify diseases or
              pests.
            </p>
          </div>
          <div className="bg-white/20 backdrop-blur-md p-6 rounded-2xl border border-white/30 transform hover:scale-105 transition-transform duration-300">
            <div className="text-4xl mb-3">💧</div>
            <h3 className="text-xl font-semibold mb-2">Resource Management</h3>
            <p className="text-blue-200 text-sm">
              Optimize watering schedules and fertilizer usage for maximum
              yield.
            </p>
          </div>
        </div>

        <button
          onClick={onEnterChat}
          className="bg-white text-sky-700 font-bold py-4 px-10 rounded-full text-lg shadow-2xl transform hover:scale-110 active:scale-100 transition-transform duration-300 ease-in-out"
        >
          Enter Smart Assistant →
        </button>
      </div>
      <footer className="absolute bottom-4 text-center text-white/70 text-sm">
        ©️ 2025 KissanSeva AI. Empowering Farmers with Technology.
      </footer>
      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 1s ease-out forwards;
        }
      `}</style>
    </div>
  );
};

// ===================================================================================
// ORIGINAL: Farmer Chatbot Component
// This is your existing chatbot code, with corrections applied.
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
  const messagesEndRef = useRef(null);
  const imageInputRef = useRef(null);
  const voiceInputRef = useRef(null);
  const textInputRef = useRef(null);
  const API_BASE = "http://localhost:5001/api";
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
  ]; // Auto-scroll to bottom

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]); // Check connection on mount
  useEffect(() => {
    checkConnection();
  }, []); // Focus text input when tab changes

  useEffect(() => {
    if (activeTab === "text" && textInputRef.current) {
      setTimeout(() => textInputRef.current.focus(), 100);
    }
  }, [activeTab]);
  const checkConnection = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      // const response = await fetch("http://localhost:5000/");
      setConnectionStatus(response.ok ? "connected" : "error");
    } catch (error) {
      setConnectionStatus("error");
      addMessage(
        "🔧 Backend server not responding. Please start the Node.js server on port 5000.",
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
  const sendTextMessage = async (messageText = null) => {
    const message = messageText || textInput.trim();
    if (!message) return;
    addMessage(message, true, "text");
    setTextInput("");
    setIsLoading(true);
    try {
      // FIXED: Used backticks (`) for template literal string
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();
      setIsLoading(false);
      if (data.success) {
        addMessage(data.response, false, "text");
      } else {
        // FIXED: Used backticks (`) for template literal string
        addMessage(`❌ ${data.error}`, false, "error");
      }
    } catch (error) {
      setIsLoading(false);
      // FIXED: Used backticks (`) for template literal string
      addMessage(`🚨 Connection failed: ${error.message}`, false, "error");
    }
  };
  const uploadImage = async () => {
    const file = imageInputRef.current?.files[0];
    if (!file) {
      addMessage("📷 Please select an image first!", false, "system");
      return;
    }
    // FIXED: Used backticks (`) for template literal string
    addMessage(`📸 Analyzing image: ${file.name}`, true, "image");
    setIsLoading(true);
    const formData = new FormData();
    formData.append("image", file);
    try {
      // FIXED: Used backticks (`) for template literal string
      const response = await fetch(`${API_BASE}/upload-image`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setIsLoading(false);
      if (data.success) {
        addMessage(data.response, false, "image");
      } else {
        // FIXED: Used backticks (`) for template literal string
        addMessage(`❌ ${data.error}`, false, "error");
      }
    } catch (error) {
      setIsLoading(false);
      // FIXED: Used backticks (`) for template literal string
      addMessage(`🚨 Upload failed: ${error.message}`, false, "error");
    }
    imageInputRef.current.value = "";
  };
  const uploadVoice = async () => {
    const file = voiceInputRef.current?.files[0];
    if (!file) {
      addMessage("🎤 Please select an audio file first!", false, "system");
      return;
    }
    // FIXED: Used backticks (`) for template literal string
    addMessage(`🗣 Processing voice: ${file.name}`, true, "voice");
    setIsLoading(true);
    const formData = new FormData();
    formData.append("voice", file);
    try {
      // FIXED: Used backticks (`) for template literal string
      const response = await fetch(`${API_BASE}/upload-voice`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setIsLoading(false);
      if (data.success) {
        addMessage(data.response, false, "voice");
      } else {
        // FIXED: Used backticks (`) for template literal string
        addMessage(`❌ ${data.error}`, false, "error");
      }
    } catch (error) {
      setIsLoading(false);
      // FIXED: Used backticks (`) for template literal string
      addMessage(`🚨 Processing failed: ${error.message}`, false, "error");
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
      return "bg-gradient-to-br from-blue-500 to-blue-600 text-white ml-auto";
    }
    const styles = {
      welcome:
        "bg-gradient-to-br from-cyan-500 to-cyan-700 border-2 border-cyan-300",
      text: "bg-gradient-to-br from-fuchsia-500 to-fuchsia-700 border-2 border-fuchsia-200",
      image:
        "bg-gradient-to-br from-purple-500 to-purple-700 border-2 border-purple-200",
      voice:
        "bg-gradient-to-br from-orange-500 to-orange-700 border-2 border-orange-200",
      system:
        "bg-gradient-to-br from-gray-500 to-gray-700 border-2 border-gray-200",
      error: "bg-gradient-to-br from-red-500 to-red-700 border-2 border-red-200",
    };
    return styles[message.type] || styles.text;
  };

  const TabButton = ({ id, icon, label, isActive, onClick }) => (
    <button
      onClick={onClick}
      className={`flex-1 flex flex-col items-center p-2 rounded-t-xl font-medium transition-all duration-300 transform ${
        isActive
          ? "bg-gray-800 text-cyan-400 shadow-lg scale-105 border-b-4 border-cyan-500"
          : "bg-gray-700 text-gray-400 hover:bg-gray-600 hover:scale-102"
      }`}
    >
      <span className="text-2xl mb-1">{icon}</span>
      <span className="text-sm">{label}</span>
    </button>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-950 to-purple-950 p-4 text-white">
      <div className="max-w-10/12 mx-auto">
        {/* Header */}
        <div className="bg-gray-800/95 backdrop-blur-md rounded-t-3xl shadow-2xl p-3 border-b-4 border-cyan-500">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="text-6xl">🌾</div>
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-fuchsia-400 bg-clip-text text-transparent">
                  KissanSeva AI
                </h1>
                <p className="text-gray-400 text-lg">
                  AI Agronomist
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 bg-gray-900 px-4 py-2 rounded-full">
              <span>{getStatusIcon()}</span>
              <span className="text-sm font-medium text-gray-400">
                {connectionStatus === "connected"
                  ? "Online"
                  : connectionStatus === "error"
                  ? "Offline"
                  : "Connecting..."}
              </span>
            </div>
          </div>
        </div>
        <div className="bg-gray-800/95 backdrop-blur-md shadow-2xl rounded-b-3xl overflow-hidden">
          {/* Tab Navigation */}
          <div className="bg-gray-900 px-6 pt-6">
            <div className="flex gap-2">
              <TabButton
                id="text"
                icon="💬"
                label="Chat"
                isActive={activeTab === "text"}
                onClick={() => setActiveTab("text")}
              />
              <TabButton
                id="image"
                icon="📸"
                label="Image Analysis"
                isActive={activeTab === "image"}
                onClick={() => setActiveTab("image")}
              />
              <TabButton
                id="voice"
                icon="🎤"
                label="Voice Message"
                isActive={activeTab === "voice"}
                onClick={() => setActiveTab("voice")}
              />
            </div>
          </div>
          <div className="flex flex-col lg:flex-row h-[600px]">
            {/* Chat Messages */}
            <div className="flex-1 flex flex-col bg-gradient-to-b from-gray-900 to-gray-800">
              <div className="flex-1 p-6 overflow-y-auto space-y-4">
                {messages.map((message) => (
                  <div key={message.id} className="flex animate-slideIn">
                    <div
                      className={`max-w-[85%] p-4 rounded-2xl shadow-lg transition-all duration-300 hover:shadow-xl ${getMessageStyle(
                        message
                      )}`}
                    >
                      <div
                        className={`text-sm font-semibold mb-2 flex items-center gap-2 ${
                          message.isUser ? "text-blue-100" : "text-gray-200"
                        }`}
                      >
                        <span className="text-lg">
                          {message.isUser
                            ? "👤"
                            : message.type === "welcome"
                            ? "🌟"
                            : "🤖"}
                        </span>
                        <span>{message.isUser ? "You" : "KissanSeva AI"}</span>
                        <span className="ml-auto text-xs opacity-70">
                          {message.timestamp.toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                      </div>
                      <div
                        className={`leading-relaxed whitespace-pre-line ${
                          message.isUser ? "text-white" : "text-gray-300"
                        }`}
                      >
                        {message.content}
                      </div>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex animate-pulse">
                    <div className="bg-gradient-to-br from-gray-900 to-gray-800 border-2 border-gray-700 p-4 rounded-2xl shadow-lg">
                      <div className="flex items-center gap-3 text-cyan-400">
                        <div className="animate-spin w-6 h-6 border-3 border-gray-700 border-t-cyan-400 rounded-full"></div>
                        <span className="font-medium">
                          KisaanSeva AI is thinking...
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
              {/* Ask Anything Input - At bottom of chat area */}
              {activeTab === "text" && (
                <div className="border-t-2 border-gray-700 bg-gray-800 p-4">
                  <div className="flex flex-col gap-3">
                    <textarea
                      ref={textInputRef}
                      value={textInput}
                      onChange={(e) => setTextInput(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="💬 Ask me anything about farming..."
                      rows="2"
                      // FIXED: Changed text color to be visible on dark background
                      className="w-full p-3 bg-gray-700 border-2 border-gray-700 text-white rounded-xl focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50 focus:outline-none transition-all resize-none placeholder-gray-400"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={() => sendTextMessage()}
                        disabled={isLoading || !textInput.trim()}
                        className="flex-1 py-2 bg-gradient-to-r from-cyan-600 to-teal-600 text-white rounded-lg font-semibold hover:from-cyan-700 hover:to-teal-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-md"
                      >
                        {isLoading ? "Sending..." : "Send ✨"}
                      </button>
                      <button
                        onClick={() => setTextInput("")}
                        className="px-4 py-2 bg-gray-700 text-gray-400 rounded-lg hover:bg-gray-600 transition-colors"
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
            {/* Input Panel */}
            <div className="lg:w-96 bg-gradient-to-b from-gray-900 to-gray-800 border-l-2 border-gray-700">
              {/* Text Input Tab */}
              {activeTab === "text" && (
                <div className="p-6 h-full flex flex-col">
                  {/* Quick Actions */}
                  <div className="mb-6">
                    <h4 className="font-semibold text-cyan-400 mb-3">
                      ⚡ Quick Actions
                    </h4>
                    <div className="grid grid-cols-2 gap-2">
                      {quickActions.map((action, index) => (
                        <button
                          key={index}
                          onClick={() => sendTextMessage(action.query)}
                          className="p-3 bg-gray-800 border-2 border-gray-700 text-gray-300 rounded-xl hover:border-cyan-400 hover:text-cyan-400 hover:bg-gray-700 transition-all duration-200 transform hover:scale-105"
                        >
                          <div className="text-2xl mb-1">{action.icon}</div>
                          <div className="text-xs font-medium text-gray-400">
                            {action.text}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                  {/* Categories */}
                  <div className="flex-1 overflow-y-auto">
                    <h4 className="font-semibold text-fuchsia-400 mb-3">
                      📚 Browse Topics
                    </h4>
                    {Object.entries(farmingCategories).map(
                      ([category, questions]) => (
                        <details key={category} className="mb-3">
                          <summary className="cursor-pointer p-3 bg-gray-800 rounded-lg border border-gray-700 hover:bg-gray-700 transition-colors capitalize font-medium text-gray-300">
                            {category} ({questions.length})
                          </summary>
                          <div className="mt-2 space-y-1">
                            {questions.map((question, index) => (
                              <button
                                key={index}
                                onClick={() => sendTextMessage(question)}
                                className="w-full text-left p-2 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded border hover:border-gray-600 transition-colors"
                              >
                                {question}
                              </button>
                            ))}
                          </div>
                        </details>
                      )
                    )}
                  </div>
                </div>
              )}
              {/* Image Input Tab */}
              {activeTab === "image" && (
                <div className="p-6 h-full flex flex-col">
                  <h3 className="text-xl font-bold text-fuchsia-400 mb-4 flex items-center gap-2">
                    <span>📸</span> Image Analysis
                  </h3>
                  <div className="flex-1 flex flex-col gap-4">
                    <div className="border-2 border-dashed border-cyan-400 rounded-xl bg-gray-900 p-6 text-center hover:border-cyan-500 hover:bg-gray-800 transition-all">
                      <input
                        ref={imageInputRef}
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files[0];
                          if (file) {
                            const preview =
                              document.getElementById("image-preview");
                            preview.textContent = `📷 ${file.name}`;
                          }
                        }}
                      />
                      <button
                        onClick={() => imageInputRef.current?.click()}
                        className="w-full"
                      >
                        <div className="text-4xl mb-2">📸</div>
                        <div className="font-semibold text-cyan-400 mb-1">
                          Upload Plant Image
                        </div>
                        <div className="text-sm text-gray-400">
                          Click to select image
                        </div>
                        <div
                          id="image-preview"
                          className="text-sm text-cyan-500 mt-2 font-medium"
                        ></div>
                      </button>
                    </div>
                    <button
                      onClick={uploadImage}
                      disabled={isLoading}
                      className="w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-semibold hover:from-purple-700 hover:to-pink-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-lg"
                    >
                      {isLoading ? "Analyzing..." : "Analyze Image 🔍"}
                    </button>
                    <div className="bg-gray-900 p-4 rounded-xl border-2 border-purple-800">
                      <h4 className="font-semibold text-purple-400 mb-2">
                        💡 Pro Tips
                      </h4>
                      <ul className="text-sm text-gray-400 space-y-1">
                        <li>• Use clear, well-lit photos</li>
                        <li>• Focus on the affected area</li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}
              {/* Voice Input Tab */}
              {activeTab === "voice" && (
                <div className="p-6 h-full flex flex-col">
                  <h3 className="text-xl font-bold text-fuchsia-400 mb-4 flex items-center gap-2">
                    <span>🎤</span> Voice Message
                  </h3>
                  <div className="flex-1 flex flex-col gap-4">
                    <div className="border-2 border-dashed border-orange-400 rounded-xl bg-gray-900 p-6 text-center hover:border-orange-500 hover:bg-gray-800 transition-all">
                      <input
                        ref={voiceInputRef}
                        type="file"
                        accept="audio/*"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files[0];
                          if (file) {
                            const preview =
                              document.getElementById("voice-preview");
                            preview.textContent = `🎵 ${file.name}`;
                          }
                        }}
                      />
                      <button
                        onClick={() => voiceInputRef.current?.click()}
                        className="w-full"
                      >
                        <div className="text-4xl mb-2">🎤</div>
                        <div className="font-semibold text-orange-400 mb-1">
                          Upload Audio File
                        </div>
                        <div className="text-sm text-gray-400">
                          Click to select audio
                        </div>
                        <div
                          id="voice-preview"
                          className="text-sm text-orange-500 mt-2 font-medium"
                        ></div>
                      </button>
                    </div>
                    <button
                      onClick={uploadVoice}
                      disabled={isLoading}
                      className="w-full py-4 bg-gradient-to-r from-orange-600 to-red-600 text-white rounded-xl font-semibold hover:from-orange-700 hover:to-red-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-lg"
                    >
                      {isLoading ? "Processing..." : "Process Voice 🗣"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
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
// NEW: Main App Component
// This component manages whether to show the LandingPage or the FarmerChatbot.
// ===================================================================================
const App = () => {
  const [showChat, setShowChat] = useState(false);

  // If showChat is true, render the chatbot, otherwise render the landing page.
  if (showChat) {
    return <FarmerChatbot />;
  }

  return <LandingPage onEnterChat={() => setShowChat(true)} />;
};

// Make sure to export the main App component
export default App;