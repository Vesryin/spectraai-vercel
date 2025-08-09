import React from 'react';
import { Sparkles } from 'lucide-react';

const TypingIndicator: React.FC = () => {
  return (
    <div className="flex items-start space-x-4">
      {/* Professional Spectra Avatar */}
      <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 via-purple-500 to-indigo-500 flex items-center justify-center text-white shadow-lg">
        <Sparkles className="w-6 h-6" />
      </div>

      {/* Professional Typing Animation */}
      <div className="bg-slate-800/50 border border-slate-700/50 backdrop-blur-sm px-6 py-4 rounded-2xl rounded-tl-md shadow-lg">
        <div className="text-xs font-semibold mb-2 text-violet-400 uppercase tracking-wider">
          Spectra AI
        </div>
        <div className="flex items-center space-x-3">
          <div className="typing-indicator flex space-x-1">
            <span className="w-2.5 h-2.5 bg-violet-400 rounded-full inline-block"></span>
            <span className="w-2.5 h-2.5 bg-purple-400 rounded-full inline-block"></span>
            <span className="w-2.5 h-2.5 bg-indigo-400 rounded-full inline-block"></span>
          </div>
          <span className="text-sm text-slate-400 font-medium">Processing...</span>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
