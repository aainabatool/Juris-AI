import React, { useState, useRef, useEffect } from "react";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf";
import "./App.css";

// Configure PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/legacy/build/pdf.worker.min.js",
  import.meta.url
).toString();

const API_BASE = "http://localhost:8000";

const App = () => {
  const [sessionId, setSessionId] = useState(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const chatEndRef = useRef(null);

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // File Upload Handler
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    setError("");

    try {
      if (file.type === "application/pdf") {
        const buffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;
        let text = "";
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          text += content.items.map((it) => it.str).join(" ") + "\n";
        }
        setInput(text);
      } else {
        const text = await file.text();
        setInput(text);
      }
    } catch (err) {
      console.error(err);
      setError("Error reading file. Please upload a valid PDF or TXT.");
    }
  };

  // Send Query
  const handleSend = async () => {
    if (!input.trim()) return setError("Enter your query or upload a document first.");

    setLoading(true);
    setError("");

    const userMessage = input.trim();
    setMessages((prev) => [...prev, { type: "user", text: userMessage }]);
    setInput("");

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage, session_id: sessionId }),
      });

      const data = await res.json();
      setSessionId(data.session_id);

      setMessages((prev) => [...prev, { type: "ai", text: data.response }]);
    } catch (err) {
      console.error(err);
      setError("Backend unreachable. Make sure FastAPI is running.");
    } finally {
      setLoading(false);
    }
  };

  // Handle Enter key
  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#0e0e0e] text-gray-100 font-sans">
      {/* Header */}
      <header className="bg-[#181818] border-b border-gray-800 p-4 text-center font-semibold text-lg text-blue-400">
        AdalAI — Pakistani Legal Intelligence Assistant
      </header>

      {/* Chat Window */}
      <main className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`max-w-3xl px-4 py-3 rounded-2xl whitespace-pre-wrap leading-relaxed ${
              msg.type === "user"
                ? "bg-blue-600 text-white self-end ml-auto"
                : "bg-[#1a1a1a] border border-gray-800 text-gray-100 self-start mr-auto"
            }`}
          >
            {msg.text}
          </div>
        ))}
        {loading && (
          <div className="text-gray-400 text-sm italic">Analyzing...</div>
        )}
        <div ref={chatEndRef} />
      </main>

      {/* Error Bar */}
      {error && (
        <div className="bg-red-600 text-white text-center py-2 text-sm">
          {error}
        </div>
      )}

      {/* Input Area */}
      <footer className="bg-[#181818] border-t border-gray-800 p-4 flex flex-col gap-3">
        {/* File Upload */}
        <div className="flex items-center gap-3">
          <label className="bg-blue-600 hover:bg-blue-500 px-3 py-1 rounded-lg text-white text-sm cursor-pointer">
            Upload PDF / TXT
            <input
              type="file"
              accept=".pdf,.txt"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
          {fileName && (
            <span className="text-gray-400 text-xs truncate max-w-[200px]">
              {fileName}
            </span>
          )}
        </div>

        {/* Textarea */}
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          rows={3}
          placeholder="Ask about a legal issue, paste a case summary, or upload a document..."
          className="w-full bg-[#121212] border border-gray-700 rounded-xl p-3 text-sm text-gray-100 focus:ring-2 focus:ring-blue-500 outline-none resize-none"
        />

        {/* Send Button */}
        <div className="flex justify-end">
          <button
            onClick={handleSend}
            disabled={loading}
            className="bg-green-600 hover:bg-green-500 disabled:bg-gray-700 px-5 py-2 rounded-xl text-sm font-medium text-white transition"
          >
            {loading ? "Processing..." : "Send"}
          </button>
        </div>
      </footer>

      {/* Footer Note */}
      <div className="text-xs text-gray-500 text-center py-2 border-t border-gray-800 bg-[#0e0e0e]">
        © 2025 AdalAI — Built for Pakistan’s Legal System
      </div>
    </div>
  );
};

export default App;
