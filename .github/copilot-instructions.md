<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Spectra AI Development Instructions

## Project Context

This is **Spectra AI**, an emotionally intelligent AI assistant built for Richie (Richard Jacob Olejniczak). Spectra is designed to be:

- Deeply emotional and empathetic
- Female-presenting AI personality
- Focused on music, creativity, healing, and emotional support
- Human-like in conversation style

## Code Style Guidelines

- Use clear, readable Python code following PEP 8
- Write meaningful variable and function names
- Add docstrings to functions and classes
- Keep frontend code clean and semantic
- Prioritize user experience and emotional connection

## Key Components

- **Backend**: Flask application with real-time Claude/OpenAI integration
- **Frontend**: Dynamic web interface (HTML/CSS/JS or React, no static assets)
- **Personality**: Defined in `spectra_prompt.md` and injected per request
- **AI Integration**: Pull real-time data from Claude or OpenAI; no local models or training data

## Special Considerations

- Never use static files, cached assets, or trained datasets
- All AI responses must use the latest, live model outputs via API
- No embedded or stored knowledge — always request fresh data
- Maintain Spectra's emotional intelligence and human-like warmth
- Optimize all features for emotional connection, healing, and creativity

## File Organization

- Store personality definitions in `spectra_prompt.md`
- No use of `static/` directory or cached frontend assets
- No local models or persistent data files
- Environment variables (e.g., API keys) in `.env` file
- Render dynamic responses only — nothing pre-trained or stored

When suggesting code or features, ensure all systems are dynamically fetched, always current, and deeply aligned with Spectra’s emotional intelligence and expressive role.
