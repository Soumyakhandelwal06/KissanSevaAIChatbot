import React, { useState, useRef, useEffect } from "react";
import "./App.css";

// ===================================================================================
// CONFIGURATION
// Use the FastAPI server URL (running on port 8000 by default)
// ===================================================================================
const API_BASE = import.meta.env.VITE_API_BASE_URL ? `${import.meta.env.VITE_API_BASE_URL}/api` : "http://localhost:8000/api";
const HEALTH_CHECK_URL = import.meta.env.VITE_API_BASE_URL ? `${import.meta.env.VITE_API_BASE_URL}/health` : "http://localhost:8000/health";

// ===================================================================================
// Error Boundary for Debugging
// ===================================================================================
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ error: error, errorInfo: errorInfo });
    console.error("Uncaught error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 bg-white text-red-800 h-screen overflow-auto">
          <h1 className="text-2xl font-bold mb-4">Something went wrong.</h1>
          <details className="whitespace-pre-wrap">
            {this.state.error && this.state.error.toString()}
            <br />
            {this.state.errorInfo && this.state.errorInfo.componentStack}
          </details>
          <button onClick={() => window.location.reload()} className="mt-4 px-4 py-2 bg-red-600 text-white rounded">
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ===================================================================================
// Landing Page Component (Futuristic Update)
// ===================================================================================
const LandingPage = ({ onEnterChat }) => {
  const [showGuide, setShowGuide] = useState(false);
  const [showResources, setShowResources] = useState(false);
  const [showTeam, setShowTeam] = useState(false);

  const TeamModal = () => (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-[#1B4332]/60 backdrop-blur-md" onClick={() => setShowTeam(false)}></div>
      <div className="harvest-card w-full max-w-lg p-8 relative z-10 animate-scale-up overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-[#FFB703] via-[#2D6A4F] to-[#FFB703]"></div>
        <button 
          onClick={() => setShowTeam(false)}
          className="absolute top-6 right-6 w-10 h-10 rounded-full bg-[#2D6A4F]/5 flex items-center justify-center text-[#2D6A4F] hover:bg-[#2D6A4F] hover:text-white transition-all font-bold"
        >✕</button>
        
        <div className="text-center mb-8">
          <span className="text-4xl mb-2 block">🤝</span>
          <h2 className="text-3xl font-black text-[#1B4332] mb-2 tracking-tight">MEET THE MAKERS</h2>
          <p className="text-[#5D4037]/70 font-bold text-sm uppercase tracking-widest">Technocrats for Agriculture</p>
        </div>

        <div className="space-y-4">
          <div className="harvest-card p-4 flex items-center gap-4 bg-[#74C69D]/10 border border-[#2D6A4F]/10">
             <div className="w-14 h-14 bg-[#2D6A4F] rounded-full flex items-center justify-center text-2xl text-white shadow-lg">👨‍💻</div>
             <div>
                <h3 className="font-black text-[#1B4332] text-lg">Soumya Khandelwal</h3>
                <p className="text-xs font-bold text-[#2D6A4F] uppercase tracking-wider">Key Contributor</p>
             </div>
          </div>
          <div className="harvest-card p-4 flex items-center gap-4 bg-[#FFB703]/10 border border-[#FFB703]/20">
             <div className="w-14 h-14 bg-[#FFB703] rounded-full flex items-center justify-center text-2xl text-[#1B4332] shadow-lg">💡</div>
             <div>
                <h3 className="font-black text-[#1B4332] text-lg">Naman Agrawal</h3>
                <p className="text-xs font-bold text-[#5D4037]/80 uppercase tracking-wider">Key Contributor</p>
             </div>
          </div>
          <div className="harvest-card p-4 flex items-center gap-4 bg-[#E9C46A]/10 border border-[#E9C46A]/20">
             <div className="w-14 h-14 bg-[#E9C46A] rounded-full flex items-center justify-center text-2xl text-[#1B4332] shadow-lg">✨</div>
             <div>
                <h3 className="font-black text-[#1B4332] text-lg">Somya Porwal</h3>
                <p className="text-xs font-bold text-[#5D4037]/80 uppercase tracking-wider">Key Contributor</p>
             </div>
          </div>
        </div>

        <div className="mt-8 text-center">
          <p className="text-[10px] font-bold text-[#5D4037]/50 uppercase tracking-widest">
            Made with ❤️ to empower Indian Farmers
          </p>
        </div>
      </div>
    </div>
  );

  const ResourcesHub = () => (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-[#1B4332]/70 backdrop-blur-md" onClick={() => setShowResources(false)}></div>
      <div className="harvest-card w-full max-w-4xl p-8 relative z-10 animate-scale-up overflow-hidden max-h-[90vh] flex flex-col">
        <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-[#FFB703] via-[#74C69D] to-[#FFB703]"></div>
        <button 
          onClick={() => setShowResources(false)}
          className="absolute top-6 right-6 w-10 h-10 rounded-full bg-[#2D6A4F]/5 flex items-center justify-center text-[#2D6A4F] hover:bg-[#2D6A4F] hover:text-white transition-all font-bold"
        >✕</button>
        
        <div className="text-center mb-8">
          <h2 className="text-3xl font-black text-[#1B4332] mb-2 tracking-tight">RESOURCES HUB</h2>
          <p className="text-[#5D4037]/70 font-bold text-sm uppercase tracking-widest">Essential Tools for Every Farmer</p>
        </div>

        <div className="flex-1 overflow-y-auto pr-2 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[
            { tag: "LIVE", icon: "📊", title: "Mandi Prices", desc: "Real-time updates on crop prices in your nearest markers (APMCs).", color: "#2D6A4F" },
            { tag: "NEW", icon: "📜", title: "Govt Schemes", desc: "Easy guides for PM-Kisan, Fasal Bima Yojana, and local subsidies.", color: "#74C69D" },
            { tag: "LIBRARY", icon: "🐛", title: "Pest Catalog", desc: "Identify common pests and diseases with organic and chemical cures.", color: "#FFB703" },
            { tag: "ADVISORY", icon: "☀️", title: "Weather Alerts", desc: "Detailed sowing and harvesting windows based on local forecasts.", color: "#1B4332" },
            { tag: "SERVICES", icon: "🚜", title: "Rent Machinery", desc: "Connect with local Custom Hiring Centers for modern equipment.", color: "#5D4037" },
            { tag: "GUIDE", icon: "💧", title: "Water Saving", desc: "Drip irrigation models and monsoon water harvesting techniques.", color: "#40916C" }
          ].map((item, i) => (
            <div key={i} className="harvest-card p-6 flex flex-col items-start gap-4 hover:border-[#74C69D] group transition-all cursor-pointer">
              <div className="flex justify-between w-full items-center">
                <div className="text-3xl grayscale group-hover:grayscale-0 transition-all">{item.icon}</div>
                <span className="text-[9px] font-black px-2 py-0.5 rounded-full bg-[#2D6A4F]/5 text-[#2D6A4F] border border-[#2D6A4F]/10">{item.tag}</span>
              </div>
              <div>
                <h3 className="font-black text-[#1B4332] text-lg leading-tight mb-2">{item.title}</h3>
                <p className="text-xs font-medium text-[#5D4037]/70 leading-relaxed">{item.desc}</p>
              </div>
              <button className="text-[10px] font-black text-[#2D6A4F] mt-auto uppercase tracking-wider group-hover:translate-x-1 transition-transform">View Details →</button>
            </div>
          ))}
        </div>

        <div className="mt-8 p-4 bg-[#74C69D]/10 rounded-2xl flex items-center justify-between border border-[#74C69D]/20">
          <div className="flex items-center gap-3">
             <span className="text-xl">📞</span>
             <p className="text-[11px] font-bold text-[#1B4332]">Kissan Helpline: <span className="text-[#2D6A4F]">1800-180-1551</span></p>
          </div>
          <button className="text-[10px] font-black bg-[#2D6A4F] text-white px-4 py-2 rounded-lg hover:scale-105 transition-transform">CALL NOW</button>
        </div>
      </div>
    </div>
  );

  const GuideModal = () => (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-[#1B4332]/60 backdrop-blur-md" onClick={() => setShowGuide(false)}></div>
      <div className="harvest-card w-full max-w-2xl p-8 relative z-10 animate-scale-up overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-[#2D6A4F] via-[#74C69D] to-[#2D6A4F]"></div>
        <button 
          onClick={() => setShowGuide(false)}
          className="absolute top-6 right-6 w-10 h-10 rounded-full bg-[#2D6A4F]/5 flex items-center justify-center text-[#2D6A4F] hover:bg-[#2D6A4F] hover:text-white transition-all font-bold"
        >✕</button>
        
        <div className="text-center mb-10">
          <h2 className="text-3xl font-black text-[#1B4332] mb-2 tracking-tight">HOW IT WORKS</h2>
          <p className="text-[#5D4037]/70 font-bold text-sm uppercase tracking-widest">Your 4-Step Path to Better Farming</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {[
            { step: "01", icon: "📍", title: "Connect", desc: "Set your farm location and snap a photo of your soil context to get started." },
            { step: "02", icon: "💬", title: "Ask", desc: "Use voice, text, or photos to tell us about your crop concerns or questions." },
            { step: "03", icon: "⚡", title: "Process", desc: "Our AI analyzes satellite data and field metrics to find the perfect solution." },
            { step: "04", icon: "📈", title: "Grow", desc: "Receive expert advice and precise predictions to maximize your harvest yield." }
          ].map((item, i) => (
            <div key={i} className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-[#2D6A4F]/10 rounded-2xl flex items-center justify-center text-2xl relative">
                {item.icon}
                <span className="absolute -top-2 -left-2 text-[10px] font-black text-[#2D6A4F] bg-white px-1 rounded">{item.step}</span>
              </div>
              <div>
                <h3 className="font-black text-[#1B4332] text-lg mb-1">{item.title}</h3>
                <p className="text-xs font-medium text-[#5D4037]/80 leading-relaxed">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>

        <button 
          onClick={() => { setShowGuide(false); onEnterChat(); }}
          className="w-full mt-10 btn-primary py-4 text-sm font-black tracking-widest flex items-center justify-center gap-3 animate-shimmer"
        >
          I'M READY TO START 🌱
        </button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#FDFBF7] text-[#2D3436] flex flex-col items-center relative overflow-hidden">
      {/* Dynamic Background Elements */}
      <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-gradient-to-br from-[#2D6A4F]/10 to-transparent rounded-full blur-[120px] -mr-96 -mt-96 animate-float"></div>
      <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-gradient-to-tr from-[#FFB703]/10 to-transparent rounded-full blur-[100px] -ml-48 -mb-48"></div>

      {/* Navigation Bar */}
      <header className="w-full max-w-7xl px-8 py-6 flex items-center justify-between z-20">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-[#2D6A4F] rounded-xl flex items-center justify-center text-xl shadow-lg">🚜</div>
          <span className="text-xl font-black text-[#1B4332] tracking-tight">KissanSeva<span className="text-[#74C69D]">AI</span></span>
        </div>
        <div className="flex items-center gap-6">
          <button onClick={() => setShowGuide(true)} className="text-sm font-bold text-[#5D4037]/70 hover:text-[#2D6A4F] transition-colors">How it works</button>
          <button onClick={() => setShowResources(true)} className="text-sm font-bold text-[#5D4037]/70 hover:text-[#2D6A4F] transition-colors">Resources</button>
          <button onClick={() => setShowTeam(true)} className="text-sm font-bold text-[#2D6A4F] hover:text-[#1B4332] transition-colors border-2 border-[#2D6A4F]/10 px-4 py-2 rounded-full hover:bg-[#2D6A4F]/5">About Us</button>
        </div>
      </header>

      {showGuide && <GuideModal />}
      {showResources && <ResourcesHub />}
      {showTeam && <TeamModal />}

      {/* Hero Section */}
      <main className="flex-1 w-full max-w-7xl px-8 flex flex-col lg:flex-row items-center justify-center gap-16 z-10 py-12">
        <div className="flex-1 text-left animate-fade-up">
          <div className="inline-flex items-center gap-2 bg-[#74C69D]/20 text-[#1B4332] px-4 py-1.5 rounded-full text-xs font-black tracking-widest uppercase mb-6">
            <span className="w-2 h-2 bg-[#2D6A4F] rounded-full animate-pulse"></span>
            Empowering Indian Farmers
          </div>
          <h1 className="text-6xl md:text-8xl font-black leading-[0.9] text-[#1B4332] mb-8">
            GROW YOUR <br />
            <span className="text-[#2D6A4F] italic">PROSPERITY</span>
          </h1>
          <p className="text-lg md:text-xl text-[#5D4037] font-medium max-w-xl leading-relaxed mb-10 opacity-90">
            Harness the power of AI to protect your crops, optimize your harvest, 
            and secure your family's future with data-driven field support.
          </p>
          <div className="flex flex-col sm:flex-row items-start gap-4">
            <button
              onClick={onEnterChat}
              className="btn-primary px-10 py-5 text-lg font-black tracking-widest flex items-center gap-4 animate-pulse-soft group"
            >
              START FARMING SMARTER
              <span className="text-2xl group-hover:translate-x-2 transition-transform">→</span>
            </button>
            <div className="flex -space-x-3 items-center ml-2 mt-4 sm:mt-0">
               {[1,2,3,4].map(i => (
                 <div key={i} className="w-10 h-10 rounded-full border-4 border-[#FDFBF7] bg-[#2D6A4F] flex items-center justify-center text-[10px] text-white font-bold">
                   {["👨‍🌾", "👩‍🌾", "🌾", "🚜"][i-1]}
                 </div>
               ))}
               <span className="pl-6 text-xs font-bold text-[#5D4037]/60">Joined by 50k+ Farmers</span>
            </div>
          </div>
        </div>

        <div className="flex-1 relative w-full lg:w-auto flex justify-center lg:justify-end animate-scale-up">
           <div className="relative z-10 grid grid-cols-2 gap-4 w-full max-w-md">
              <div 
                className="harvest-card p-6 flex flex-col gap-4 bg-white/80 backdrop-blur-sm rotate-[-2deg]"
              >
                 <div className="w-12 h-12 bg-[#74C69D]/20 rounded-2xl flex items-center justify-center text-2xl">🌱</div>
                 <h3 className="font-black text-[#1B4332] leading-tight text-lg">Yield <br/>Predictor</h3>
                 <div className="h-1.5 w-full bg-[#2D6A4F]/10 rounded-full overflow-hidden">
                    <div className="h-full bg-[#2D6A4F] w-[85%] animate-pulse"></div>
                 </div>
              </div>
              <div 
                className="harvest-card p-6 flex flex-col gap-4 bg-white/80 backdrop-blur-sm translate-y-8 rotate-[2deg]"
              >
                 <div className="w-12 h-12 bg-[#FFB703]/20 rounded-2xl flex items-center justify-center text-2xl">🔍</div>
                 <h3 className="font-black text-[#1B4332] leading-tight text-lg">Instant <br/>Diagnosis</h3>
                 <p className="text-[10px] font-bold text-[#5D4037]/60">98% Accuracy in Pest Identification</p>
              </div>
              <div 
                className="harvest-card p-6 flex flex-col gap-4 bg-white/80 backdrop-blur-sm -translate-y-4"
              >
                 <div className="w-12 h-12 bg-[#2D6A4F]/20 rounded-2xl flex items-center justify-center text-2xl">🎤</div>
                 <h3 className="font-black text-[#1B4332] leading-tight text-lg">Voice <br/>Assistant</h3>
                 <div className="flex gap-1">
                    {[1,2,3,4,5].map(i => <div key={i} className="w-1 h-3 bg-[#2D6A4F]/40 rounded-full animate-bounce" style={{animationDelay: `${i*0.1}s`}}></div>)}
                 </div>
              </div>
              <div 
                className="harvest-card p-6 flex flex-col gap-4 bg-white/80 backdrop-blur-sm rotate-[1deg]"
              >
                 <div className="text-3xl">🌾</div>
                 <h3 className="font-black text-[#1B4332] leading-tight text-lg">Healthy <br/>Harvest</h3>
                 <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-[#2D6A4F] rounded-full animate-pulse"></div>
                    <span className="text-[10px] font-bold text-[#5D4037]/80">94% Optimal Growth</span>
                 </div>
                 <button className="text-[10px] font-black uppercase tracking-widest bg-[#2D6A4F] text-white py-2.5 rounded-lg opacity-80 cursor-default">Check Status</button>
              </div>
           </div>
           {/* Abstract Decorative Element */}
           <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] bg-[#2D6A4F]/5 rounded-[40px] -z-10 rotate-12"></div>
        </div>
      </main>

      {/* Feature Ticker */}
      <div className="w-full bg-[#1B4332] py-4 overflow-hidden whitespace-nowrap z-20">
        <div className="flex animate-scroll whitespace-nowrap gap-12 text-[#74C69D] font-black text-xs uppercase tracking-[0.2em]">
           <span>• Real-time Pest Analysis • Weather-based Irrigation • Market Price Forecasting • Multilingual AI Support • Soil Health Monitoring • Expert Farming Tips • 24/7 Field Assistance •</span>
           <span>• Real-time Pest Analysis • Weather-based Irrigation • Market Price Forecasting • Multilingual AI Support • Soil Health Monitoring • Expert Farming Tips • 24/7 Field Assistance •</span>
        </div>
      </div>

      <footer className="w-full py-6 px-8 flex flex-col md:flex-row items-center justify-between text-[#5D4037]/50 text-[10px] font-bold uppercase tracking-widest bg-[#FDFBF7] z-20">
        <div>© 2026 KissanSeva AI. All Rights Reserved.</div>
        <div className="flex gap-8 mt-4 md:mt-0">
          <a href="#" className="hover:text-[#2D6A4F] transition-colors">Privacy Policy</a>
          <a href="#" className="hover:text-[#2D6A4F] transition-colors">Terms of Service</a>
          <a href="#" className="hover:text-[#2D6A4F] transition-colors">Contact Support</a>
        </div>
      </footer>
    </div>
  );
};

const TabButton = ({ id, icon, label, isActive, onClick }) => (
  <button
    onClick={onClick}
    className={`flex items-center justify-center gap-2 px-6 py-3 rounded-2xl font-bold transition-all duration-300 ${
      isActive
        ? "bg-[#2D6A4F] text-white shadow-lg organic-shadow"
        : "text-[#5D4037] hover:bg-[#2D6A4F]/5"
    }`}
  >
    <span className="text-2xl">{icon}</span>
    <span className="hidden xl:block uppercase tracking-wider text-xs">
      {label}
    </span>
  </button>
);

// ===================================================================================
// Farmer Chatbot Component
// ===================================================================================

const FarmerChatbot = ({ initialMessage }) => {
  const [showUtility, setShowUtility] = useState(false);
  const [utilityType, setUtilityType] = useState(null); // 'language', 'settings', 'help', 'profile'
  const [language, setLanguage] = useState('en'); // 'en', 'hi'
  const [appSettings, setAppSettings] = useState({
    highContrast: false,
    voiceResponses: false,
    notifications: true
  });
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [profileData, setProfileData] = useState({
    name: "Aditya Kumar",
    location: "Village: Kanjari, Bihar",
    crop: "Rice / Paddy",
    land: "2.5 Acres"
  });
  const [messages, setMessages] = useState([
    {
      id: 1,
      content:
        initialMessage 
          ? `Searching for: ${initialMessage}...` 
          : "Namaste! 🌱 I am your KissanSeva Assistant.\n\nI can help you grow better crops, identify pests from photos, and provide weather-based advice. How can I help you today?",
      isUser: false,
      isUser: false,
      type: initialMessage ? "initial" : "welcome",
      timestamp: new Date(),
    },
  ]);

  // Update welcome message if language changes manually (Simplified for demo)
  useEffect(() => {
    // Only adapt welcome message if it is actually a welcome message (not initial query)
    if (messages.length === 1 && messages[0].type === "welcome") {
      const welcomeEn = "Namaste! 🌱 I am your KissanSeva Assistant.\n\nI can help you grow better crops, identify pests from photos, and provide weather-based advice. How can I help you today?";
      const welcomeHi = "नमस्ते! 🌱 मैं आपका किसानसेवा सहायक हूँ।\n\nमैं आपको बेहतर फसल उगाने, तस्वीरों से कीटों की पहचान करने और मौसम आधारित सलाह देने में मदद कर सकता हूँ। आज मैं आपकी क्या मदद कर सकता हूँ?";
      
      setMessages([{
        ...messages[0],
        content: language === 'hi' ? welcomeHi : welcomeEn
      }]);
    }
  }, [language]);
  const [activeTab, setActiveTab] = useState("text");
  const [textInput, setTextInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("checking");
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const imageInputRef = useRef(null);
  const voiceInputRef = useRef(null);
  const textInputRef = useRef(null);

  const [context, setContext] = useState({
    crop: "Rice",
    location: "Kerala",
    season: "Kharif",
  });

  const [predictionFeatures, setPredictionFeatures] = useState({
    N: 90, P: 42, K: 43,
    temperature: 28.5,
    humidity: 80.0,
    ph: 6.5,
    rainfall: 200.0,
  });


  const quickActions = [
    { icon: "🔍", text: "Identify Pest", query: "Can you help me identify a pest from a photo?" },
    { icon: "☀️", text: "Weather Info", query: "What's the weather forecast for my location?" },
    { icon: "🌾", text: "Crop Suggestions", query: "Based on my soil, what's the best crop to plant?" },
    { icon: "💧", text: "Irrigation Tip", query: "How much water do my crops need right now?" },
  ];

  /* Auto-scroll disabled per user request */
  const scrollToBottom = () => {
    // messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    checkConnection();
    if (initialMessage) {
      setTimeout(() => {
        sendTextMessage(initialMessage);
      }, 500);
    }
  }, []);

  const checkConnection = async () => {
    try {
      const response = await fetch(HEALTH_CHECK_URL);
      setConnectionStatus(response.ok ? "connected" : "error");
    } catch (error) {
      setConnectionStatus("error");
    }
  };

  const addMessage = (content, isUser, type = "text") => {
    setMessages((prev) => [...prev, {
      id: Date.now() + Math.random(),
      content, isUser, type,
      timestamp: new Date(),
    }]);
  };

  const renderUtilityModal = () => {
    if (!showUtility) return null;

    const views = {
      language: {
        title: language === 'en' ? "Language Selection" : "भाषा का चयन",
        icon: "🌐",
        content: (
          <div className="grid grid-cols-2 gap-4">
            {[
              { label: "English", code: "en" },
              { label: "हिन्दी (Hindi)", code: "hi" },
              { label: "ਪੰਜਾਬੀ (Punjabi)", code: "pa" },
              { label: "తెలుగు (Telugu)", code: "te" }
            ].map(lang => (
              <button 
                key={lang.code} 
                onClick={() => { setLanguage(lang.code); setShowUtility(false); }} 
                className={`farmer-input p-4 text-sm font-bold transition-all border-2 ${language === lang.code ? "border-[#2D6A4F] bg-[#2D6A4F]/10" : "border-transparent hover:bg-[#2D6A4F]/5"} text-[#1B4332]`}
              >
                {lang.label}
              </button>
            ))}
          </div>
        )
      },
      settings: {
        title: language === 'en' ? "App Settings" : "ऐप सेटिंग्स",
        icon: "⚙️",
        content: (
          <div className="space-y-4">
            {[
              { id: "highContrast", label: "High Contrast Mode", icon: "👁️" },
              { id: "voiceResponses", label: "Voice Responses", icon: "🔊" },
              { id: "notifications", label: "Notification Alerts", icon: "🔔" }
            ].map(setting => (
              <div 
                key={setting.id} 
                onClick={() => setAppSettings(prev => ({ ...prev, [setting.id]: !prev[setting.id] }))}
                className="flex items-center justify-between p-3 farmer-input cursor-pointer hover:bg-[#2D6A4F]/5 transition-colors"
              >
                <span className="text-xs font-bold text-[#1B4332] flex items-center gap-3">
                  <span className="text-base">{setting.icon}</span> {setting.label}
                </span>
                <div className={`w-10 h-5 rounded-full relative transition-colors ${appSettings[setting.id] ? "bg-[#2D6A4F]" : "bg-[#2D6A4F]/20"}`}>
                  <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${appSettings[setting.id] ? "left-6" : "left-1"}`}></div>
                </div>
              </div>
            ))}
          </div>
        )
      },
      help: {
        title: "Help & FAQ",
        icon: "❓",
        content: (
          <div className="space-y-3">
            {[
              "How to identify a pest?",
              "Setting farm location",
              "Understanding N-P-K values",
              "How to get weather alerts?"
            ].map(q => (
              <button key={q} onClick={() => { setShowUtility(false); sendTextMessage(q); }} className="w-full text-left p-3 farmer-input text-[10px] font-bold text-[#1B4332] hover:bg-[#2D6A4F]/5">
                {q} →
              </button>
            ))}
          </div>
        )
      },
      profile: {
        title: "Farmer Profile",
        icon: "👤",
        content: (
          <div className="space-y-4 text-center">
            <div className="w-20 h-20 bg-[#2D6A4F]/10 rounded-full mx-auto flex items-center justify-center text-4xl shadow-inner border-2 border-[#2D6A4F]/20">👨‍🌾</div>
            
            {isEditingProfile ? (
              <div className="space-y-3 animate-fade-up">
                <input 
                  value={profileData.name}
                  onChange={(e) => setProfileData({...profileData, name: e.target.value})}
                  className="w-full text-center font-black text-[#1B4332] bg-transparent border-b-2 border-[#2D6A4F]/20 focus:border-[#2D6A4F] outline-none pb-1"
                />
                <input 
                  value={profileData.location}
                  onChange={(e) => setProfileData({...profileData, location: e.target.value})}
                  className="w-full text-center text-[10px] font-bold text-[#5D4037]/60 uppercase tracking-widest bg-transparent border-b-2 border-[#2D6A4F]/20 focus:border-[#2D6A4F] outline-none pb-1"
                />
                <div className="grid grid-cols-2 gap-2 pt-2">
                  <div className="farmer-input p-2">
                    <span className="block text-[8px] text-[#5D4037]/40 font-black">CROP</span>
                    <input 
                      value={profileData.crop}
                      onChange={(e) => setProfileData({...profileData, crop: e.target.value})}
                      className="w-full text-center text-[10px] font-bold text-[#1B4332] bg-transparent outline-none"
                    />
                  </div>
                  <div className="farmer-input p-2">
                    <span className="block text-[8px] text-[#5D4037]/40 font-black">LAND</span>
                    <input 
                      value={profileData.land}
                      onChange={(e) => setProfileData({...profileData, land: e.target.value})}
                      className="w-full text-center text-[10px] font-bold text-[#1B4332] bg-transparent outline-none"
                    />
                  </div>
                </div>
                <button 
                  onClick={() => setIsEditingProfile(false)}
                  className="w-full py-2 bg-[#2D6A4F] text-white text-xs font-black rounded-lg hover:bg-[#1B4332] transition-colors"
                >
                  SAVE CHANGES
                </button>
              </div>
            ) : (
              <div className="animate-fade-up">
                <div>
                  <h3 className="text-lg font-black text-[#1B4332]">{profileData.name}</h3>
                  <p className="text-[10px] font-bold text-[#5D4037]/60 uppercase tracking-widest">{profileData.location}</p>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-4">
                  <div className="farmer-input p-2"><span className="block text-[8px] text-[#5D4037]/40 font-black">CROP</span><span className="text-[10px] font-bold text-[#1B4332]">{profileData.crop}</span></div>
                  <div className="farmer-input p-2"><span className="block text-[8px] text-[#5D4037]/40 font-black">LAND</span><span className="text-[10px] font-bold text-[#1B4332]">{profileData.land}</span></div>
                </div>
                <button 
                  onClick={() => setIsEditingProfile(true)}
                  className="mt-4 text-[10px] font-bold text-[#2D6A4F] hover:underline"
                >
                  EDIT PROFILE
                </button>
              </div>
            )}
          </div>
        )
      }
    };

    const view = views[utilityType];
    if (!view) return null;

    return (
      <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
        <div className="absolute inset-0 bg-[#1B4332]/40 backdrop-blur-md" onClick={() => setShowUtility(false)}></div>
        <div className="relative w-full max-w-sm harvest-card bg-[#FDFBF7] p-8 shadow-2xl animate-scale-up border-2 border-[#2D6A4F]/20">
          <button onClick={() => setShowUtility(false)} className="absolute top-4 right-4 w-10 h-10 flex items-center justify-center text-[#5D4037]/40 hover:text-[#5D4037] text-2xl font-black">×</button>
          
          <div className="flex flex-col items-center mb-8">
            <div className="w-16 h-16 bg-[#2D6A4F]/10 rounded-2xl flex items-center justify-center text-3xl mb-4 border border-[#2D6A4F]/10">
              {view.icon}
            </div>
            <h2 className="text-xl font-black text-[#1B4332] tracking-tight uppercase">{view.title}</h2>
            <div className="w-12 h-1 bg-[#2D6A4F]/20 rounded-full mt-2"></div>
          </div>

          <div className="animate-fade-up">
            {view.content}
          </div>

          <button onClick={() => setShowUtility(false)} className="w-full mt-8 btn-primary py-4 text-xs font-black tracking-widest uppercase shadow-lg">CLOSE</button>
        </div>
      </div>
    );
  };

  const sendTextMessage = async (messageText = null) => {
    const message = messageText || textInput.trim();
    if (!message) return;

    addMessage(message, true, "text");
    setTextInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: message,
          context: { ...context, features: predictionFeatures },
        }),
      });

      const data = await response.json();
      setIsLoading(false);

      if (response.ok) {
        addMessage(data.answer || "I'm here to help!", false, "text");
      } else {
        addMessage("Sorry, I'm having trouble connecting to the field office.", false, "error");
      }
    } catch (error) {
      setIsLoading(false);
      addMessage("Connection failed. Please check your internet.", false, "error");
    }
  };

  const uploadImage = async () => {
    const file = imageInputRef.current?.files[0];
    if (!file) return;

    addMessage(`Checking photo: ${file.name}`, true, "image");
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/image`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setIsLoading(false);

      if (response.ok) {
        let result = `Diagnosis: ${data.label}\n\nSolution: ${data.remedy}`;
        addMessage(result, false, "image");
      } else {
        addMessage("Could not analyze the photo. Please try again with a clearer shot.", false, "error");
      }
    } catch (error) {
      setIsLoading(false);
      addMessage("Failed to upload the photo.", false, "error");
    }
  };

  const uploadVoice = async () => {
    const file = voiceInputRef.current?.files[0];
    if (!file) return;
    addMessage("Processing your voice message...", true, "voice");
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      addMessage("I heard you! How can I help with your crops today?", false, "voice");
    }, 1500);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage();
    }
  };

  return (
    <div className="h-screen bg-[#FDFBF7] text-[#2D3436] font-sans selection:bg-[#2D6A4F]/20 overflow-hidden flex flex-col">
      <div className="max-w-full mx-auto w-full h-full flex flex-col p-2 md:p-3 gap-3">
        <header className="harvest-card p-3 flex items-center justify-between mb-1">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-[#2D6A4F] rounded-xl flex items-center justify-center text-xl organic-shadow">
              🚜
            </div>
            <div>
              <h1 className="text-xl font-black text-[#1B4332] leading-none">
                KissanSeva<span className="text-[#74C69D]">AI</span>
              </h1>
              <div className="flex items-center gap-1.5 text-[9px] font-bold text-[#5D4037]/60 mt-1">
                <span className="w-1.5 h-1.5 bg-[#74C69D] rounded-full animate-pulse"></span>
                <span>Active Field Support</span>
              </div>
            </div>
          </div>

          <nav className="flex items-center gap-1 bg-[#2D6A4F]/5 p-1 rounded-2xl mx-4">
            <TabButton id="text" icon="💬" label="Chat Advice" isActive={activeTab === "text"} onClick={() => setActiveTab("text")} />
            <TabButton id="image" icon="📸" label="Plant Scan" isActive={activeTab === "image"} onClick={() => setActiveTab("image")} />
            <TabButton id="voice" icon="🎤" label="Voice Helper" isActive={activeTab === "voice"} onClick={() => setActiveTab("voice")} />
            <TabButton id="predict" icon="📊" label="Harvest Plan" isActive={activeTab === "predict"} onClick={() => setActiveTab("predict")} />
          </nav>

          <div className="flex items-center gap-3">
             <div className="hidden sm:flex items-center gap-2 mr-4">
                {/* Utility icons removed as per user request */}
             </div>
            <div className="hidden lg:block text-right border-l border-[#2D6A4F]/10 pl-4">
              <div className="text-[8px] uppercase tracking-widest text-[#5D4037]/50 font-bold mb-0.5">Status</div>
              <div className={`text-[10px] font-bold flex items-center gap-1.5 ${connectionStatus === "connected" ? "text-[#2D6A4F]" : "text-red-500"}`}>
                {connectionStatus === "connected" ? "CONNECTED" : "OFFLINE"}
                <div className={`w-1.5 h-1.5 rounded-full ${connectionStatus === "connected" ? "bg-[#74C69D]" : "bg-red-500"}`}></div>
              </div>
            </div>
          </div>
        </header>

        <main className="flex-1 flex gap-3 min-h-0">
          <aside className="hidden md:flex w-72 flex-col gap-3 animate-fade-up">
            <section className="harvest-card flex-1 flex flex-col overflow-hidden">
              <div className="p-3 border-b border-[#2D6A4F]/10 bg-[#2D6A4F]/5">
                <h3 className="text-[10px] uppercase tracking-wider text-[#1B4332] font-black flex items-center gap-2">🚜 Farm Dashboard</h3>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {[
                  { label: "Current Crop", value: context.crop, key: "crop", icon: "🌱" },
                  { label: "Farm Location", value: context.location, key: "location", icon: "📍" },
                  { label: "Current Season", value: context.season, key: "season", icon: "⏳" },
                ].map((item) => (
                  <div key={item.key} className="space-y-1">
                    <label className="text-[9px] font-black text-[#5D4037]/70 uppercase tracking-tighter">{item.label}</label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sm">{item.icon}</span>
                      <input
                        type="text"
                        value={item.value}
                        onChange={(e) => setContext((prev) => ({ ...prev, [item.key]: e.target.value }))}
                        className="farmer-input w-full py-1.5 pl-9 pr-3 text-xs font-bold text-[#1B4332] h-9"
                      />
                    </div>
                  </div>
                ))}

                <div className="pt-3 border-t border-[#2D6A4F]/10">
                  <h4 className="text-[9px] uppercase tracking-widest text-[#5D4037]/60 font-black mb-2">Soil Nutrients</h4>
                  <div className="grid grid-cols-3 gap-2">
                    {["N", "P", "K"].map((nutrient) => (
                      <div key={nutrient} className="farmer-input p-1.5 text-center">
                        <span className="text-[8px] font-black text-[#5D4037]/40 block mb-0.5">{nutrient}</span>
                        <input
                          type="number"
                          value={predictionFeatures[nutrient]}
                          onChange={(e) => setPredictionFeatures(p => ({...p, [nutrient]: parseFloat(e.target.value)}))}
                          className="w-full bg-transparent text-center font-bold text-[#2D6A4F] outline-none text-[10px]"
                        />
                      </div>
                    ))}
                  </div>
                </div>

                <div className="pt-3 border-t border-[#2D6A4F]/10">
                  <h4 className="text-[9px] uppercase tracking-widest text-[#5D4037]/60 font-black mb-2">Soil Environment</h4>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      {l:"Temp",k:"temperature"}, 
                      {l:"Humid",k:"humidity"},
                      {l:"pH",k:"ph"},
                      {l:"Rain",k:"rainfall"}
                    ].map(f => (
                      <div key={f.k} className="farmer-input p-1.5">
                        <span className="text-[8px] font-black text-[#5D4037]/50 uppercase">{f.l}</span>
                        <input
                          type="number"
                          value={predictionFeatures[f.k]}
                          onChange={(e) => setPredictionFeatures(p => ({...p, [f.k]: parseFloat(e.target.value)}))}
                          className="w-full bg-transparent font-bold text-[#1B4332] outline-none text-[10px]"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          </aside>

          <section className="flex-1 harvest-card flex flex-col min-h-0 relative overflow-hidden animate-fade-up">
            <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 bg-white/30 scroll-smooth">
              {messages.map((message) => (
                <div key={message.id} className={`flex ${message.isUser ? "justify-end" : "justify-start"}`}>
                  <div className={`max-w-[85%] p-4 rounded-[22px] shadow-sm ${message.isUser ? "user-bubble rounded-tr-none" : "ai-bubble rounded-tl-none"}`}>
                    <div className="flex items-center gap-2 mb-1.5 opacity-60">
                      <span className="text-[10px] font-black uppercase tracking-wider">{message.isUser ? "You" : "KissanSeva Assistant"}</span>
                    </div>
                    <div className="text-sm leading-relaxed whitespace-pre-wrap font-medium">
                      {message.content}
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="ai-bubble p-4 rounded-[22px] rounded-tl-none italic text-xs text-[#5D4037]/60 flex items-center gap-3">
                    <div className="flex gap-1">
                      <div className="w-1 h-1 bg-[#74C69D] rounded-full animate-bounce"></div>
                      <div className="w-1 h-1 bg-[#74C69D] rounded-full animate-bounce [animation-delay:0.2s]"></div>
                      <div className="w-1 h-1 bg-[#74C69D] rounded-full animate-bounce [animation-delay:0.4s]"></div>
                    </div>
                    Field advisor is processing...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <footer className="p-4 bg-white border-t border-[#2D6A4F]/10">
              {activeTab === "text" && (
                <div className="flex items-center gap-3">
                  <textarea
                    ref={textInputRef}
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    onKeyDown={handleKeyPress}
                    placeholder="Type your question for KissanSeva..."
                    className="farmer-input flex-1 py-3 px-5 text-sm font-bold resize-none h-12"
                    rows="1"
                  />
                  <button onClick={() => sendTextMessage()} disabled={isLoading || !textInput.trim()} className="btn-primary w-12 h-12 flex items-center justify-center organic-shadow">
                    <span className="text-xl">🌱</span>
                  </button>
                </div>
              )}
              {activeTab === "image" && (
                <div className="flex flex-col items-center py-6 bg-[#2D6A4F]/5 rounded-2xl border-2 border-dashed border-[#2D6A4F]/20">
                  <span className="text-4xl mb-3">📸</span>
                  <p className="text-[10px] font-bold text-[#5D4037]/70 uppercase mb-4">Upload Crop Photo for Diagnosis</p>
                  <label className="btn-primary px-8 py-3 cursor-pointer text-xs font-black tracking-widest">
                    CHOOSE FILE
                    <input type="file" ref={imageInputRef} onChange={uploadImage} accept="image/*" className="hidden" />
                  </label>
                </div>
              )}
              {activeTab === "voice" && (
                <div className="flex flex-col items-center py-6 bg-[#FFB703]/5 rounded-2xl border-2 border-dashed border-[#FFB703]/20">
                  <span className="text-4xl mb-3">🎤</span>
                  <p className="text-[10px] font-bold text-[#5D4037]/70 uppercase mb-4">Voice Assistant Activity</p>
                  <label className="bg-[#FFB703] text-[#1B4332] font-black px-8 py-3 rounded-full cursor-pointer text-xs tracking-widest organic-shadow">
                    START TALKING
                    <input type="file" ref={voiceInputRef} onChange={uploadVoice} accept="audio/*" className="hidden" />
                  </label>
                </div>
              )}
              {activeTab === "predict" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 animate-fade-up">
                  <div className="harvest-card p-5 border border-[#2D6A4F]/10 flex flex-col justify-center">
                    <div className="text-3xl mb-2">📈</div>
                    <h3 className="text-lg font-black text-[#1B4332] mb-1">Harvest Forecast</h3>
                    <p className="text-[10px] text-[#5D4037]/70 font-bold leading-relaxed">
                      AI yield prediction based on your current soil metrics and weather.
                    </p>
                  </div>
                  <div className="flex flex-col gap-3">
                    <div className="harvest-card p-3 bg-[#74C69D]/5 border border-[#2D6A4F]/10">
                        <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                             <div className="flex justify-between text-[8px] font-black tracking-tighter"><span className="text-[#5D4037]/60">N-P-K:</span> <span className="text-[#2D6A4F]">{predictionFeatures.N}-{predictionFeatures.P}-{predictionFeatures.K}</span></div>
                             <div className="flex justify-between text-[8px] font-black tracking-tighter"><span className="text-[#5D4037]/60">TEMP:</span> <span className="text-[#2D6A4F]">{predictionFeatures.temperature}°C</span></div>
                             <div className="flex justify-between text-[8px] font-black tracking-tighter"><span className="text-[#5D4037]/60">HUMID:</span> <span className="text-[#2D6A4F]">{predictionFeatures.humidity}%</span></div>
                             <div className="flex justify-between text-[8px] font-black tracking-tighter"><span className="text-[#5D4037]/60">pH:</span> <span className="text-[#2D6A4F]">{predictionFeatures.ph}</span></div>
                        </div>
                    </div>
                    <button
                      onClick={() => sendTextMessage("Predict yield based on my soil data.")}
                      className="btn-primary py-4 text-xs tracking-widest font-black"
                    >
                      GENERATE REPORT ⚡
                    </button>
                  </div>
                </div>
              )}
            </footer>
          </section>
          
          <aside className="hidden lg:flex w-16 flex-col gap-2 animate-fade-up">
            {quickActions.map((action, i) => (
              <button
                key={i}
                onClick={() => sendTextMessage(action.query)}
                title={action.text}
                className="w-full h-16 harvest-card flex items-center justify-center text-2xl hover:bg-[#2D6A4F] hover:text-white transition-all organic-shadow group border border-[#2D6A4F]/10"
              >
                <span className="group-hover:scale-125 transition-transform">{action.icon}</span>
              </button>
            ))}
            <div className="flex-1"></div>
            <button 
              onClick={() => { setUtilityType('settings'); setShowUtility(true); }}
              className="w-full h-16 harvest-card flex items-center justify-center text-xl hover:bg-[#FFB703] transition-all border border-[#2D6A4F]/10"
            >
              ⚙️
            </button>
          </aside>

          {renderUtilityModal()}
        </main>
      </div>
    </div>
  );
};

// ===================================================================================
// Main App Component
// ===================================================================================
const App = () => {
  const [showChat, setShowChat] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [initialMsg, setInitialMsg] = useState(null);

  const handleEnterChat = (msg = null) => {
    // Ensure we only pass strings, not React event objects
    setInitialMsg(typeof msg === 'string' ? msg : null);
    setIsInitializing(true);
    setTimeout(() => {
      setShowChat(true);
      setIsInitializing(false);
    }, 1200);
  };

  if (isInitializing) {
    return (
      <div className="h-screen bg-[#FDFBF7] flex flex-col items-center justify-center text-[#1B4332]">
        <div className="relative mb-8 animate-leaf">
          <div className="text-9xl filter drop-shadow-2xl">🚜</div>
        </div>
        <h2 className="text-2xl font-black tracking-widest text-[#2D6A4F] mb-6 animate-pulse">
          PREPARING ASSISTANT...
        </h2>
        <div className="w-64 h-2 bg-[#2D6A4F]/10 rounded-full overflow-hidden">
          <div className="h-full bg-[#2D6A4F] animate-pulse w-full"></div>
        </div>
      </div>
    );
  }

  return showChat ? (
    <ErrorBoundary>
      <FarmerChatbot initialMessage={initialMsg} />
    </ErrorBoundary>
  ) : (
    <LandingPage onEnterChat={handleEnterChat} />
  );
};

// Make sure to export the main App component
export default App;
