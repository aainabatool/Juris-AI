import React, { useState, useRef, useEffect } from "react";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf";
import "./App.css"; // optional for extra styling

// Configure PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/legacy/build/pdf.worker.min.js",
  import.meta.url
).toString();

const API_BASE = "http://localhost:8000";

const App = () => {
  const [caseText, setCaseText] = useState("");
  const [messages, setMessages] = useState([]); // chat history
  const [loading, setLoading] = useState(false);
  const [selectedOption, setSelectedOption] = useState("analyze");
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const chatEndRef = useRef(null);

  // Scroll to bottom on new message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle file upload
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    setError("");
    setCaseText("");

    try {
      if (file.type === "application/pdf") {
        const arrayBuffer = await file.arrayBuffer();
        const pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        let text = "";
        for (let i = 1; i <= pdfDoc.numPages; i++) {
          const page = await pdfDoc.getPage(i);
          const content = await page.getTextContent();
          text += content.items.map((s) => s.str).join(" ") + "\n";
        }
        setCaseText(text);
      } else {
        const text = await file.text();
        setCaseText(text);
      }
    } catch (err) {
      console.error(err);
      setError("Error reading file. Please upload a valid PDF or TXT.");
    }
  };

  // Send message to backend
  const handleSend = async () => {
    if (!caseText.trim()) {
      setError("Please enter or upload a case first.");
      return;
    }

    setLoading(true);
    setError("");

    // Add user message
    setMessages((prev) => [...prev, { type: "user", text: caseText }]);
    setCaseText("");

    try {
      const res = await fetch(`${API_BASE}/${selectedOption}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: caseText }),
      });

      const data = await res.json();
      let reply = "";

      if (selectedOption === "classify") reply = data.classification_result;
      else if (selectedOption === "priority") reply = data.priority_result;
      else reply = `Classification:\n${data.classification}\n\nPriority:\n${data.priority}`;

      setMessages((prev) => [...prev, { type: "ai", text: reply }]);
    } catch (err) {
      console.error(err);
      setError("Could not reach backend. Is FastAPI running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#0f0f0f] text-gray-100">
      {/* Header */}
      <div className="bg-[#1a1a1a] p-4 shadow-md text-center text-blue-400 font-bold text-xl">
        AdalAI - Pakistani Legal Case Assistant
      </div>

      {/* Chat container */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`max-w-[70%] p-3 rounded-lg whitespace-pre-wrap ${
              msg.type === "user" ? "bg-blue-600 self-end text-white" : "bg-gray-800 self-start text-gray-100"
            }`}
          >
            {msg.text}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-600 text-white text-center p-2">{error}</div>
      )}

      {/* Input area */}
      <div className="bg-[#1a1a1a] p-4 flex flex-col gap-2 border-t border-gray-700">
        {/* File Upload */}
        <div className="flex gap-2 items-center">
          <label className="bg-blue-600 hover:bg-blue-500 px-3 py-1 rounded-lg cursor-pointer text-white font-semibold">
            Upload PDF or TXT
            <input
              type="file"
              accept=".pdf,.txt"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
          {fileName && <span className="text-gray-400">{fileName}</span>}
        </div>

        {/* Case Input */}
        <textarea
          value={caseText}
          onChange={(e) => setCaseText(e.target.value)}
          rows={3}
          placeholder="Enter or paste your case text here..."
          className="w-full bg-[#121212] border border-gray-700 rounded-lg p-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />

        {/* Options & Send */}
        <div className="flex gap-2 mt-2">
          <select
            value={selectedOption}
            onChange={(e) => setSelectedOption(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-100"
          >
            <option value="classify">Classify</option>
            <option value="priority">Priority</option>
            <option value="analyze">Full Analysis</option>
          </select>
          <button
            onClick={handleSend}
            disabled={loading}
            className="bg-green-600 hover:bg-green-500 px-4 py-2 rounded-lg text-white font-semibold disabled:bg-gray-700"
          >
            {loading ? "Analyzing..." : "Send"}
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="text-xs text-gray-500 text-center p-2">
        © 2025 AdalAI | Made for Pakistani Law System
      </div>
    </div>
  );
};

export default App;
