import React from 'react';
import { Message } from '../types';
import { User, Sparkles, AlertCircle } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isSpectra = message.sender === 'spectra';
  
  const formatTime = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className={`flex items-start space-x-4 ${isSpectra ? '' : 'flex-row-reverse space-x-reverse'}`}>
      {/* Professional Avatar */}
      <div className={`flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center shadow-lg ${
        isSpectra 
          ? 'bg-gradient-to-br from-violet-500 via-purple-500 to-indigo-500 text-white' 
          : 'bg-gradient-to-br from-slate-600 to-slate-700 text-slate-300'
      }`}>
        {isSpectra ? (
          <Sparkles className="w-6 h-6" />
        ) : (
          <User className="w-6 h-6" />
        )}
      </div>

      {/* Message Content */}
      <div className={`max-w-2xl ${isSpectra ? '' : 'text-right'}`}>
        {/* Sender Label */}
        <div className={`text-xs font-semibold mb-2 uppercase tracking-wider ${
          isSpectra 
            ? message.isError 
              ? 'text-red-400' 
              : 'text-violet-400'
            : 'text-slate-400'
        }`}>
          {isSpectra ? 'Spectra AI' : 'You'}
        </div>

        {/* Message Bubble */}
        <div className={`px-6 py-4 rounded-2xl shadow-lg border backdrop-blur-sm ${
          isSpectra
            ? message.isError
              ? 'bg-red-500/10 border-red-500/20 text-red-300'
              : 'bg-slate-800/50 border-slate-700/50 text-slate-100'
            : 'bg-gradient-to-br from-violet-500/90 to-purple-600/90 border-violet-400/20 text-white'
        } ${isSpectra ? 'rounded-tl-md' : 'rounded-tr-md'}`}>
          
          {/* Error Icon */}
          {message.isError && (
            <div className="flex items-center space-x-2 mb-2">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <span className="text-xs font-medium text-red-400 uppercase tracking-wide">
                Connection Error
              </span>
            </div>
          )}

          {/* Message Text */}
          <div className="text-base leading-relaxed whitespace-pre-wrap font-medium">
            {message.content}
          </div>
        </div>

        {/* Timestamp */}
        <div className={`text-xs text-slate-500 mt-2 font-medium ${
          isSpectra ? 'text-left' : 'text-right'
        }`}>
          {formatTime(message.timestamp)}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
