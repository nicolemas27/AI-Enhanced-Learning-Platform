// Store quiz-related styles
const quizStyles = `
  #yt-quiz-container {
    position: fixed;
    right: 20px;
    top: 80px;
    background: white;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    width: 360px;
    max-height: calc(100vh - 100px);
    overflow-y: auto;
    z-index: 9999;
    font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    border: 1px solid #e5e5e5;
    font-size: 14px;
    line-height: 1.5;
    color: #030303;
    scrollbar-width: thin;
    scrollbar-color: #888 #f1f1f1;
  }

  #yt-quiz-container .question {
    background: #f8f9fa;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 16px;
    border: 1px solid #e5e5e5;
  }

  #yt-quiz-container .question p {
    margin-bottom: 12px;
  }

  #yt-quiz-container .options label {
    display: block;
    padding: 12px 16px;
    margin: 8px 0;
    cursor: pointer;
    border-radius: 8px;
    background: white;
    border: 1px solid #e5e5e5;
    transition: all 0.2s;
  }

  #yt-quiz-container .options label:hover {
    border-color: #1a73e8;
    background: #f8f9fa;
  }

  #yt-quiz-container button {
    width: 100%;
    padding: 12px 20px;
    background: #ff0000;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 500;
    margin-top: 16px;
    transition: all 0.2s;
  }

  #yt-quiz-container button:hover {
    background: #d90000;
  }

  #yt-quiz-container .results {
    margin-top: 24px;
    padding: 16px;
    background: #f8f9fa;
    border-radius: 12px;
    border: 1px solid #e5e5e5;
  }

  #yt-quiz-container .results h3 {
    margin-bottom: 16px;
    font-size: 16px;
  }

  #yt-quiz-container .feedback > div {
    background: white;
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 8px;
    border: 1px solid #e5e5e5;
  }

  #yt-quiz-container .correct-answer {
    color: #2e7d32;
    font-weight: 500;
  }

  #yt-quiz-container .incorrect-answer {
    color: #d32f2f;
    font-weight: 500;
  }

  #yt-quiz-container::-webkit-scrollbar {
    width: 8px;
  }

  #yt-quiz-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
  }

  #yt-quiz-container::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
  }

  #yt-quiz-container::-webkit-scrollbar-thumb:hover {
    background: #555;
  }
`;

let currentQuiz = null;

// Initialize popup
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const { geminiApiKey } = await chrome.storage.local.get('geminiApiKey');
    const apiSection = document.getElementById('api-section');
    if (apiSection) {
      if (!geminiApiKey) {
        apiSection.classList.add('show');
      }
    }

    // Settings button
    const settingsButton = document.getElementById('show-settings');
    if (settingsButton) {
      settingsButton.addEventListener('click', () => {
        const apiSection = document.getElementById('api-section');
        if (apiSection) {
          apiSection.classList.toggle('show');
        }
      });
    }

    // Save API key button
    const saveKeyButton = document.getElementById('save-key');
    if (saveKeyButton) {
      saveKeyButton.addEventListener('click', async () => {
        const apiKeyInput = document.getElementById('api-key');
        const apiKey = apiKeyInput?.value.trim();
        
        if (!apiKey) {
          showError('Please enter an API key');
          return;
        }
        
        await chrome.storage.local.set({ geminiApiKey: apiKey });
        showError('', false);
        showSuccess('API key saved successfully!');
        
        const apiSection = document.getElementById('api-section');
        if (apiSection) {
          apiSection.classList.remove('show');
        }
      });
    }

    // Generate quiz button
    const generateButton = document.getElementById('generate-quiz');
    if (generateButton) {
      generateButton.addEventListener('click', generateQuiz);
    }
  } catch (error) {
    console.error('Initialization error:', error);
  }
});

async function generateQuiz() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  if (!tab.url.includes('youtube.com/watch')) {
    showError('Please navigate to a YouTube video first');
    return;
  }
  
  const { geminiApiKey } = await chrome.storage.local.get('geminiApiKey');
  if (!geminiApiKey) {
    showError('Please save your Gemini API key first');
    const apiSection = document.getElementById('api-section');
    if (apiSection) {
      apiSection.classList.add('show');
    }
    return;
  }
  
  setLoading(true);
  showError('', false);
  
  try {
    // Inject content script first
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['content.js']
    });

    // Then execute the quiz generation
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: async (apiKey, styles) => {
        // Add styles to the page
        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);

        // Create or get quiz container
        let quizContainer = document.getElementById('yt-quiz-container');
        if (!quizContainer) {
          quizContainer = document.createElement('div');
          quizContainer.id = 'yt-quiz-container';
          document.body.appendChild(quizContainer);
        }
        
        try {
          const transcript = await getTranscript();
          if (!transcript) {
            return { error: 'Could not find video content' };
          }
          
          const videoTitle = document.querySelector('h1.ytd-video-primary-info-renderer')?.textContent?.trim();
          const questions = await generateQuestionsWithGemini(transcript, videoTitle, apiKey);
          
          return { questions };
        } catch (error) {
          return { error: error.message };
        }
      },
      args: [geminiApiKey, quizStyles]
    });
    
    const quiz = results[0].result;
    if (quiz.error) {
      showError(quiz.error);
      return;
    }
    
    currentQuiz = quiz;
    displayQuiz(quiz);
  } catch (error) {
    showError('Failed to generate quiz: ' + error.message);
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  const loadingEl = document.getElementById('loading');
  const generateButton = document.getElementById('generate-quiz');
  if (loadingEl) loadingEl.classList.toggle('active', isLoading);
  if (generateButton) generateButton.disabled = isLoading;
}

function showError(message, show = true) {
  const error = document.getElementById('error');
  if (error) {
    error.textContent = message;
    error.style.display = show ? 'block' : 'none';
  }
}

function showSuccess(message) {
  const error = document.getElementById('error');
  if (error) {
    error.textContent = message;
    error.style.color = '#4caf50';
    error.style.display = 'block';
    setTimeout(() => {
      error.style.display = 'none';
      error.style.color = '#d32f2f';
    }, 3000);
  }
}

function displayQuiz(quiz) {
  const container = document.getElementById('quiz-container');
  if (!container) return;

  container.innerHTML = quiz.questions.map((q, i) => `
    <div class="question">
      <p><strong>Q${i + 1}:</strong> ${q.question}</p>
      <div class="options">
        ${q.options.map((opt, j) => `
          <label>
            <input type="radio" name="q${i}" value="${j}">
            ${opt}
          </label>
        `).join('')}
      </div>
    </div>
  `).join('') + '<button id="submit-quiz">Submit Quiz</button>';
  
  const submitButton = document.getElementById('submit-quiz');
  if (submitButton) {
    submitButton.addEventListener('click', checkAnswers);
  }
}

function checkAnswers() {
  if (!currentQuiz) return;
  
  const container = document.getElementById('quiz-container');
  if (!container) return;

  let score = 0;
  const feedback = [];
  
  currentQuiz.questions.forEach((q, i) => {
    const selected = document.querySelector(`input[name="q${i}"]:checked`)?.value;
    if (selected === undefined) {
      feedback.push(`
        <div>
          <p>Question ${i + 1}: Not answered</p>
          <p class="correct-answer">Correct answer: ${q.options[q.correct]}</p>
        </div>
      `);
    } else {
      const correct = selected == q.correct;
      score += correct ? 1 : 0;
      feedback.push(`
        <div>
          <p>Question ${i + 1}: <span class="${correct ? 'correct-answer' : 'incorrect-answer'}">
            ${correct ? 'Correct' : 'Incorrect'}
          </span></p>
          <p>Your answer: ${q.options[selected]}</p>
          ${!correct ? `<p class="correct-answer">Correct answer: ${q.options[q.correct]}</p>` : ''}
        </div>
      `);
    }
  });
  
  container.innerHTML += `
    <div class="results">
      <h3>Results</h3>
      <p>Score: ${score}/${currentQuiz.questions.length}</p>
      <div class="feedback">${feedback.join('')}</div>
    </div>
  `;
}
