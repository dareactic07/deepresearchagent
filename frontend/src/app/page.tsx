"use client";

import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Plus, Search, Trash2, ArrowRight, Loader2, Library, User } from 'lucide-react';

const API_BASE = "http://localhost:8000/api";

export default function DeepResearchUI() {
  const [sessions, setSessions] = useState<any[]>([]);
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [messages, setMessages] = useState<any[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [inputValue]);

  // Initial load
  useEffect(() => {
    fetchSessions();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current && activeSession) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth"
      });
    }
  }, [messages, isProcessing, activeSession]);

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      const data = await res.json();
      setSessions(data);
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  };

  const loadSession = async (id: string) => {
    if (activeSession === id) return;
    setActiveSession(id);
    setIsProcessing(false);
    setMessages([]); // Clear while loading
    try {
      const res = await fetch(`${API_BASE}/sessions/${id}`);
      const data = await res.json();
      setMessages(data.messages || []);
    } catch (e) {
      console.error("Session load error", e);
      setMessages([]);
    }
  };

  const startNewChat = () => {
    setActiveSession(null);
    setMessages([]);
    setInputValue("");
  };

  const deleteSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await fetch(`${API_BASE}/sessions/${id}`, { method: "DELETE" });
      if (activeSession === id) {
        startNewChat();
      }
      fetchSessions();
    } catch (err) {
      console.error("Delete failed", err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isProcessing) return;

    const query = inputValue.trim();
    setInputValue("");
    
    // Add user message optimistically
    const userMsg = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setIsProcessing(true);

    try {
      if (!activeSession) {
        // Deep Research
        const res = await fetch(`${API_BASE}/research`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ topic: query })
        });
        const data = await res.json();
        await fetchSessions();
        await loadSession(data.session_id);
      } else {
        // RAG Chat
        const res = await fetch(`${API_BASE}/sessions/${activeSession}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: query })
        });
        const data = await res.json();
        setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
      }
    } catch (err) {
      console.error("API Error", err);
      setMessages((prev) => [...prev, { role: "assistant", content: "⚠️ Network Error. Ensure the FastAPI backend is running on port 8000." }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#191A1A] font-sans selection:bg-blue-500/30">
      
      {/* Sidebar */}
      <div className="w-[260px] bg-[#202222] border-r border-[#2E3030] flex flex-col h-full flex-shrink-0 transition-all">
        <div className="p-4 pt-5 pb-2">
          <div className="flex items-center gap-2 px-2 text-[#E8E8E8] font-semibold text-lg hover:opacity-80 cursor-pointer transition-opacity" onClick={startNewChat}>
            <Search className="w-5 h-5 text-blue-400" />
            <span className="tracking-tight">Research</span>
          </div>
        </div>
        
        <div className="px-3 mt-4 mb-2">
          <button 
            onClick={startNewChat}
            className="w-full flex items-center justify-between text-[#E8E8E8] hover:bg-[#2E3030]/60 py-2.5 px-3 rounded-lg transition-colors font-medium border border-[#3E4040]/50 shadow-sm"
          >
            <span className="text-[15px]">New Thread</span>
            <Plus size={18} className="text-zinc-400" />
          </button>
        </div>

        <div className="px-4 py-2 mt-4 text-xs font-semibold text-[#8B8D8D] uppercase tracking-wider">
          Library
        </div>

        <div className="flex-1 overflow-y-auto px-2 pb-4 space-y-0.5 custom-scrollbar">
          {sessions.length === 0 && (
            <div className="text-sm text-[#8B8D8D] px-4 py-3">No threads yet</div>
          )}
          {sessions.map((session) => (
            <div 
              key={session.id}
              onClick={() => loadSession(session.id)}
              className={`group flex items-center justify-between px-3 py-2.5 rounded-md cursor-pointer transition-all text-[14.5px] ${
                activeSession === session.id 
                  ? 'bg-[#2E3030] text-[#FFFFFF] shadow-sm font-medium' 
                  : 'hover:bg-[#2E3030]/40 text-[#B4B6B6]'
              }`}
            >
              <div className="flex items-center gap-2.5 overflow-hidden w-full">
                 <span className="truncate pr-2">{session.topic}</span>
              </div>
              <button 
                onClick={(e) => deleteSession(session.id, e)}
                className="opacity-0 group-hover:opacity-100 hover:bg-[#3E4040] p-1 rounded-md transition-all ease-in-out text-[#8B8D8D] hover:text-red-400 -mr-1"
                title="Delete thread"
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
        
        {/* Bottom Profile Anchor */}
        <div className="p-4 border-t border-[#2E3030]">
          <div className="flex items-center gap-3 px-2 py-2 text-sm font-medium text-[#B4B6B6] hover:text-white cursor-pointer transition-colors">
            <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs">U</div>
            User
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full relative">
        
        {/* State 1: New Chat Centered View */}
        {!activeSession && messages.length === 0 && !isProcessing && (
          <div className="flex-1 flex flex-col items-center justify-center -mt-20 px-4 sm:px-8">
            <h1 className="text-3xl sm:text-4xl font-normal text-[#E8E8E8] mb-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
              Where knowledge begins
            </h1>
            
            <div className="w-full max-w-[720px] mx-auto animate-in fade-in slide-in-from-bottom-5 duration-700 delay-100">
              <form 
                onSubmit={handleSubmit} 
                className="bg-[#202222] border border-[#3E4040] focus-within:border-[#5C5F5F] hover:border-[#4B4D4D] rounded-[24px] shadow-2xl transition-all flex flex-col pt-3 pb-3 px-4 relative Group"
              >
                <textarea
                  ref={textareaRef}
                  className="w-full bg-transparent outline-none text-[#F5F5F5] text-lg resize-none placeholder-[#7B7D7D] min-h-[44px] max-h-[30vh] overflow-y-auto mb-10 pb-0"
                  placeholder="Ask anything..."
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
                <div className="absolute bottom-3 right-4 flex items-center justify-end">
                  <button 
                    type="submit" 
                    disabled={!inputValue.trim()}
                    className={`flex items-center justify-center w-8 h-8 rounded-full transition-all ${
                      inputValue.trim() ? 'bg-white text-black hover:bg-zinc-200' : 'bg-[#2E3030] text-[#7B7D7D]'
                    }`}
                  >
                    <ArrowRight size={18} strokeWidth={2.5} />
                  </button>
                </div>
              </form>
            </div>
            
            <div className="mt-8 flex gap-3 animate-in fade-in duration-1000 delay-300">
                <span className="text-xs font-semibold text-[#666868] uppercase tracking-wider flex items-center gap-1.5 bg-[#202222] px-3 py-1.5 rounded-full border border-[#2E3030] shadow-sm"><Search size={12}/> Pro Search enabled</span>
            </div>
          </div>
        )}

        {/* State 2: Active Chat View */}
        {(activeSession || messages.length > 0 || isProcessing) && (
          <>
            <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 sm:px-10 pb-40 pt-10 scroll-smooth">
              <div className="max-w-[800px] mx-auto flex flex-col gap-8">
                {messages.map((msg, idx) => (
                  <div key={idx} className="flex flex-col gap-3 animate-in fade-in slide-in-from-bottom-2 duration-500">
                    
                    {msg.role === 'user' ? (
                      <div className="text-2xl sm:text-3xl font-semibold text-[#E8E8E8] tracking-tight leading-tight pt-6">
                        {msg.content}
                      </div>
                    ) : (
                      <div className="flex gap-4">
                        <div className="flex-shrink-0 mt-1">
                          <div className="w-8 h-8 rounded-full bg-[#E8E8E8] flex items-center justify-center">
                            <Search className="text-[#191A1A]" size={16} strokeWidth={2.5} />
                          </div>
                        </div>
                        <div className="flex-1 w-full text-[1.05rem] text-[#D4D4D4] markdown break-words max-w-full overflow-hidden">
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {/* Loading State Skeleton */}
                {isProcessing && (
                  <div className="flex gap-4 animate-in fade-in duration-300">
                     <div className="flex-shrink-0">
                          <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center border border-blue-500/30">
                            <Loader2 className="text-blue-400 animate-spin" size={16} />
                          </div>
                      </div>
                    <div className="flex-1 w-full space-y-4 pt-1">
                      <div className="h-4 bg-[#2A2D2D] rounded animate-pulse w-3/4"></div>
                      <div className="h-4 bg-[#2A2D2D] rounded animate-pulse w-5/6"></div>
                      <div className="h-4 bg-[#2A2D2D] rounded animate-pulse w-1/2"></div>
                      {!activeSession && (
                        <div className="mt-4 p-4 rounded-xl border border-[#2E3030] bg-[#202222]/50 text-sm text-[#8B8D8D] flex items-center gap-3">
                            <Library size={16} /> 
                            <span>Initiating comprehensive web scrape and synthesis... (1-3 min)</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Sticky Bottom Input Area */}
            <div className="absolute flex justify-center w-full bottom-0 bg-gradient-to-t from-[#191A1A] via-[#191A1A] to-transparent pt-10 pb-6 pointer-events-none px-4 sm:px-10">
              <div className="w-full max-w-[800px] pointer-events-auto shadow-2xl rounded-[24px]">
                <form 
                  onSubmit={handleSubmit} 
                  className="bg-[#202222] border border-[#3E4040] focus-within:border-[#5C5F5F] rounded-[24px] shadow-lg transition-all flex flex-col pt-2.5 pb-2.5 px-4 relative"
                >
                  <textarea
                    ref={textareaRef}
                    className="w-full bg-transparent outline-none text-[#F5F5F5] text-[15px] resize-none placeholder-[#7B7D7D] min-h-[24px] max-h-[150px] overflow-y-auto mb-8 pb-0"
                    placeholder="Ask a follow up..."
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={isProcessing}
                  />
                  <div className="absolute bottom-2 right-3 flex items-center justify-end">
                    <button 
                      type="submit" 
                      disabled={isProcessing || !inputValue.trim()}
                      className={`flex items-center justify-center w-[30px] h-[30px] rounded-full transition-all ${
                        inputValue.trim() && !isProcessing ? 'bg-black text-[#E8E8E8] border border-[#4B4D4D] hover:bg-[#2E3030]' : 'bg-transparent text-[#5C5F5F]'
                      }`}
                    >
                      {isProcessing ? <Loader2 className="animate-spin" size={16} /> : <ArrowRight size={16} />}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </>
        )}

      </div>
    </div>
  );
}
