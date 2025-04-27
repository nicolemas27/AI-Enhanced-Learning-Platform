// Validation function
async function validateVideo() {
    // Check if video is available
    const videoElement = document.querySelector('video');
    if (!videoElement) {
        throw new Error('No video found on this page');
    }

    // Check if video is loaded
    if (videoElement.readyState === 0) {
        throw new Error('Video is not loaded yet');
    }

    // Check for video title
    const titleElement = document.querySelector('h1.ytd-video-primary-info-renderer');
    if (!titleElement || !titleElement.textContent.trim()) {
        throw new Error('Could not find video title');
    }

    return true;
}

async function getTranscript() {
    // First try the "..." menu method
    try {
        // Click the "..." button
        const moreButton = await waitForElement('button.ytp-button[aria-label="More actions"]');
        if (moreButton) {
            moreButton.click();
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        // Look for "Show transcript" in the menu
        const menuItems = document.querySelectorAll('tp-yt-paper-item');
        const transcriptButton = Array.from(menuItems)
            .find(el => el.textContent.toLowerCase().includes('show transcript'));
        
        if (transcriptButton) {
            transcriptButton.click();
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    } catch (error) {
        console.log('First method failed, trying alternative...');
    }

    // Try alternative method - direct transcript button
    try {
        const transcriptButton = await waitForElement('button[aria-label="Show transcript"]');
        if (transcriptButton) {
            transcriptButton.click();
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    } catch (error) {
        console.log('Alternative method failed...');
    }

    // Wait for transcript panel to appear and get text
    try {
        // Try multiple selectors for transcript text
        const selectors = [
            'ytd-transcript-segment-renderer',
            'ytd-transcript-segment-list-renderer',
            '.segment-text',
            '.cue-group',
            '.ytd-transcript-body-renderer'
        ];

        let transcriptText = '';
        for (const selector of selectors) {
            const elements = await waitForElements(selector, 2000);
            if (elements && elements.length > 0) {
                transcriptText = Array.from(elements)
                    .map(el => el.textContent.trim())
                    .join(' ');
                if (transcriptText) break;
            }
        }

        if (!transcriptText) {
            // If no transcript, try extracting from video description and title
            const title = document.querySelector('h1.ytd-video-primary-info-renderer')?.textContent || '';
            const description = document.querySelector('ytd-expander#description')?.textContent || '';
            
            if (title || description) {
                return `Title: ${title}\n\nDescription: ${description}`;
            }
            
            throw new Error('No transcript or description found');
        }

        return transcriptText;
    } catch (error) {
        throw new Error('Could not find transcript. Please ensure the video has captions enabled.');
    }
}

// Helper function to wait for an element
function waitForElement(selector, timeout = 5000) {
    return new Promise((resolve, reject) => {
        const element = document.querySelector(selector);
        if (element) {
            resolve(element);
            return;
        }

        const observer = new MutationObserver((mutations, obs) => {
            const element = document.querySelector(selector);
            if (element) {
                obs.disconnect();
                resolve(element);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        setTimeout(() => {
            observer.disconnect();
            reject(new Error(`Timeout waiting for ${selector}`));
        }, timeout);
    });
}

// Helper function to wait for multiple elements
function waitForElements(selector, timeout = 5000) {
    return new Promise((resolve, reject) => {
        const elements = document.querySelectorAll(selector);
        if (elements.length > 0) {
            resolve(elements);
            return;
        }

        const observer = new MutationObserver((mutations, obs) => {
            const elements = document.querySelectorAll(selector);
            if (elements.length > 0) {
                obs.disconnect();
                resolve(elements);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        setTimeout(() => {
            observer.disconnect();
            resolve(document.querySelectorAll(selector)); // Resolve with whatever we found
        }, timeout);
    });
}

async function generateQuestionsWithGemini(transcript, videoTitle, apiKey) {
    const API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent';

    // Truncate transcript if too long
    const maxLength = 4000;
    const truncatedTranscript = transcript.length > maxLength 
        ? transcript.slice(0, maxLength) + '...' 
        : transcript;

    const prompt = `
        Generate exactly 5 multiple choice questions based on this YouTube video content.
        Format your response as a JSON array that strictly follows this structure:
        [
            {
                "question": "Specific question based on the video content?",
                "options": [
                    "First option with the correct answer",
                    "Second incorrect option",
                    "Third incorrect option",
                    "Fourth incorrect option"
                ],
                "correct": 0
            }
        ]

        Rules:
        1. Use actual content from the video, not generic questions
        2. Make sure each question has exactly 4 options
        3. Make the incorrect options plausible but clearly wrong
        4. The "correct" field must be the index (0-3) of the correct answer
        5. Return ONLY the JSON array, no additional text or explanations
        6. Options should be complete phrases, not placeholders
        7. Questions should test understanding of specific details from the video

        Video Title: ${videoTitle}
        Video Content: ${truncatedTranscript}
    `;

    try {
        const response = await fetch(`${API_URL}?key=${apiKey}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents: [{
                    parts: [{ text: prompt }]
                }],
                generationConfig: {
                    temperature: 0.7,
                    topK: 40,
                    topP: 0.95,
                    maxOutputTokens: 1024,
                }
            })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        
        try {
            const questionsText = data.candidates[0].content.parts[0].text;
            // Clean up the response to ensure it's valid JSON
            const cleanedText = questionsText.trim().replace(/```json\n?|\n?```/g, '');
            const questions = JSON.parse(cleanedText);

            // Validate the response format
            if (!Array.isArray(questions) || questions.length !== 5) {
                throw new Error('Invalid response format: expected array of 5 questions');
            }

            // Validate each question
            questions.forEach((q, i) => {
                if (!q.question || !Array.isArray(q.options) || q.options.length !== 4 || typeof q.correct !== 'number') {
                    throw new Error(`Invalid question format at index ${i}`);
                }
            });

            return questions;
        } catch (error) {
            console.error('Parse error:', error);
            // Fallback response if parsing fails
            return [
                {
                    question: "What is the main topic of this video?",
                    options: [
                        videoTitle,
                        "Unrelated topic 1",
                        "Unrelated topic 2",
                        "Unrelated topic 3"
                    ],
                    correct: 0
                }
            ];
        }
    } catch (error) {
        throw new Error('Failed to generate questions: ' + error.message);
    }
}

async function generateQuizContent(apiKey) {
    try {
        await validateVideo();
        const transcript = await getTranscript();
        if (!transcript) {
            return { error: 'Could not find video content' };
        }
        
        const videoTitle = document.querySelector('h1.ytd-video-primary-info-renderer')?.textContent?.trim();
        const questions = await generateQuestionsWithGemini(transcript, videoTitle, apiKey);
        
        // Validate questions before returning
        if (!questions || !Array.isArray(questions) || questions.length === 0) {
            return { error: 'Failed to generate valid questions' };
        }
        
        return { questions };
    } catch (error) {
        return { error: error.message };
    }
}
