import React, { useState, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import * as chatAPI from '../../services/chatAPI';
import styles from './ChatWindow.module.css';
import { SYSTEM_PROMPT } from './systemPrompt';

/**
 * ChatWindow Component
 * LLM-style chat interface for AI music production assistant
 */
const ChatWindow = () => {
  const { state } = useApp();
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your AI music production assistant. I can help you with composition suggestions, technical advice, and creative ideas for your project. What would you like to work on?',
      timestamp: new Date().toISOString()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Build DAW context for the API
  const buildDAWContext = () => {
    return {
      bpm: state.bpm || 120,
      key: state.generationParams?.key || 'C',
      isBPMMode: state.isBPMMode || false,
      totalDuration: state.totalDuration || 30,
      trackCount: state.buses?.reduce((total, bus) => total + bus.tracks.length, 0) || 0,
      buses: state.buses?.map(bus => ({
        id: bus.id,
        name: bus.name,
        type: bus.type,
        trackCount: bus.tracks.length
      })) || []
    };
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);

    try {
      // Call chat API
      const response = await chatAPI.sendChatMessage({
        system_prompt: SYSTEM_PROMPT,
        daw_context: buildDAWContext(),
        message: inputMessage,
        conversation_history: messages
      });

      // Add assistant response to chat
      const assistantMessage = {
        role: 'assistant',
        content: response.message,
        timestamp: response.timestamp
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Chat error:', err);
      setError(err.message || 'Failed to get response from AI assistant');
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: 'Chat cleared. How can I help you with your music production?',
        timestamp: new Date().toISOString()
      }
    ]);
    setError(null);
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className={styles.chatWindow}>
      <div className={styles.header}>
        <div className={styles.headerTitle}>
          <i className="fa-solid fa-robot"></i>
          <span>AI Music Assistant</span>
        </div>
        <button className={styles.clearButton} onClick={clearChat} title="Clear chat">
          <i className="fa-solid fa-trash"></i>
        </button>
      </div>

      <div className={styles.messagesContainer}>
        {messages.map((message, index) => (
          <div
            key={index}
            className={`${styles.message} ${
              message.role === 'user' ? styles.userMessage : styles.assistantMessage
            }`}
          >
            <div className={styles.messageIcon}>
              {message.role === 'user' ? (
                <i className="fa-solid fa-user"></i>
              ) : (
                <i className="fa-solid fa-robot"></i>
              )}
            </div>
            <div className={styles.messageContent}>
              <div className={styles.messageText}>{message.content}</div>
              <div className={styles.messageTime}>
                {formatTimestamp(message.timestamp)}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className={`${styles.message} ${styles.assistantMessage}`}>
            <div className={styles.messageIcon}>
              <i className="fa-solid fa-robot"></i>
            </div>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className={styles.errorMessage}>
            <i className="fa-solid fa-exclamation-triangle"></i>
            <span>{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className={styles.inputContainer}>
        <textarea
          ref={inputRef}
          className={styles.input}
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask me about composition, production, or creative ideas..."
          rows={1}
          disabled={isLoading}
        />
        <button
          className={styles.sendButton}
          onClick={handleSendMessage}
          disabled={!inputMessage.trim() || isLoading}
        >
          <i className="fa-solid fa-paper-plane"></i>
        </button>
      </div>
    </div>
  );
};

export default ChatWindow;
