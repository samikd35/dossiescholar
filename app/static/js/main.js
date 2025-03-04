document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const navItems = document.querySelectorAll('nav ul li');
    const tabContents = document.querySelectorAll('.tab-content');
    const agentCards = document.querySelectorAll('.agent-card');
    const queryInput = document.getElementById('query-input');
    const searchButton = document.getElementById('search-button');
    const backToSearchButton = document.getElementById('back-to-search');
    const searchContainer = document.querySelector('.search-container');
    const resultsContainer = document.querySelector('.results-container');
    const loadingSpinner = document.querySelector('.loading-spinner');
    const resultContent = document.getElementById('result-content');
    const sourcesList = document.getElementById('sources-list');
    const historyList = document.getElementById('history-list');
    const detailLevel = document.getElementById('detail-level');
    const maxResults = document.getElementById('max-results');
    
    // State
    let selectedAgentType = 'summarization';
    let searchHistory = JSON.parse(localStorage.getItem('searchHistory')) || [];
    
    // Initialize
    renderSearchHistory();
    
    // Tab Navigation
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.getAttribute('data-tab');
            
            // Update active tab
            navItems.forEach(navItem => navItem.classList.remove('active'));
            item.classList.add('active');
            
            // Show selected tab content
            tabContents.forEach(content => {
                if (content.id === `${tabId}-tab`) {
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                }
            });
        });
    });
    
    // Agent Selection
    agentCards.forEach(card => {
        card.addEventListener('click', () => {
            agentCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            selectedAgentType = card.getAttribute('data-agent');
            
            // Update max results based on agent type
            if (selectedAgentType === 'summarization') {
                updateMaxResultsOptions([3, 5, 10, 15]);
                maxResults.value = '5';
            } else if (selectedAgentType === 'literature_review') {
                updateMaxResultsOptions([5, 10, 15, 20]);
                maxResults.value = '10';
            } else if (selectedAgentType === 'deep_research') {
                updateMaxResultsOptions([5, 10, 15, 20, 25]);
                maxResults.value = '12';
            }
        });
    });
    
    // Update max results options
    function updateMaxResultsOptions(values) {
        maxResults.innerHTML = '';
        values.forEach(value => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            maxResults.appendChild(option);
        });
    }
    
    // Search Function
    searchButton.addEventListener('click', performSearch);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    
    async function performSearch() {
        const query = queryInput.value.trim();
        
        if (!query) {
            alert('Please enter a research query');
            return;
        }
        
        // Show loading state
        searchContainer.style.display = 'none';
        resultsContainer.style.display = 'block';
        loadingSpinner.style.display = 'flex';
        resultContent.innerHTML = '';
        sourcesList.innerHTML = '';
        
        try {
            // Make API request
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    agent_type: selectedAgentType,
                    detail_level: detailLevel.value,
                    max_results: parseInt(maxResults.value)
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to get results');
            }
            
            const data = await response.json();
            
            // Add to search history
            addToSearchHistory(query, selectedAgentType);
            
            // Display results
            renderResults(data);
        } catch (error) {
            console.error('Error:', error);
            resultContent.innerHTML = `
                <div class="error-message">
                    <h3>Error</h3>
                    <p>Sorry, an error occurred while processing your request. Please try again.</p>
                    <p class="error-details">${error.message}</p>
                </div>
            `;
        } finally {
            loadingSpinner.style.display = 'none';
        }
    }
    
    // Render Results
    function renderResults(data) {
        // Format and display main result
        resultContent.innerHTML = formatMarkdown(data.result);
        
        // Display sources
        sourcesList.innerHTML = '';
        if (data.sources && data.sources.length > 0) {
            data.sources.forEach(source => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                
                const title = source.metadata.title || 'Unknown Title';
                const authors = source.metadata.authors || 'Unknown Authors';
                const year = source.metadata.year || 'Unknown Year';
                const score = source.relevance_score || 0;
                
                sourceItem.innerHTML = `
                    <div class="source-title">${title}</div>
                    <div class="source-authors">${authors}</div>
                    <div class="source-year">${year}</div>
                    <div class="source-score">Relevance: ${score}</div>
                `;
                
                sourcesList.appendChild(sourceItem);
            });
        } else {
            sourcesList.innerHTML = '<p class="empty-message">No sources available</p>';
        }
    }
    
    // Format markdown-like text to HTML
    function formatMarkdown(text) {
        if (!text) return '';
        
        // Replace headers
        text = text.replace(/^### (.*$)/gm, '<h3>$1</h3>');
        text = text.replace(/^## (.*$)/gm, '<h2>$1</h2>');
        text = text.replace(/^# (.*$)/gm, '<h1>$1</h1>');
        
        // Replace bold and italic
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Replace lists
        text = text.replace(/^\s*\d+\.\s+(.*$)/gm, '<li>$1</li>');
        text = text.replace(/^\s*\-\s+(.*$)/gm, '<li>$1</li>');
        
        // Add paragraph tags
        text = text.replace(/^(?!<[hl]|<li).*$/gm, function(m) {
            return (m.trim() === '') ? '' : '<p>' + m + '</p>';
        });
        
        // Wrap lists
        text = text.replace(/<li>.*?<\/li>/gs, function(m) {
            return '<ul>' + m + '</ul>';
        });
        text = text.replace(/<\/ul>\s*<ul>/g, '');
        
        return text;
    }
    
    // Handle back to search
    backToSearchButton.addEventListener('click', () => {
        resultsContainer.style.display = 'none';
        searchContainer.style.display = 'block';
    });
    
    // Search History Functions
    function addToSearchHistory(query, agentType) {
        const historyItem = {
            id: Date.now(),
            query: query,
            agentType: agentType,
            date: new Date().toISOString()
        };
        
        searchHistory.unshift(historyItem);
        
        // Keep only the last 10 searches
        if (searchHistory.length > 10) {
            searchHistory = searchHistory.slice(0, 10);
        }
        
        // Save to localStorage
        localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
        
        // Re-render history
        renderSearchHistory();
    }
    
    function renderSearchHistory() {
        if (searchHistory.length === 0) {
            historyList.innerHTML = '<p class="empty-message">No history yet. Start by searching for research papers.</p>';
            return;
        }
        
        historyList.innerHTML = '';
        searchHistory.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            
            // Format date
            const date = new Date(item.date);
            const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            
            // Format agent type
            let agentLabel = item.agentType;
            if (item.agentType === 'summarization') {
                agentLabel = 'Summary';
            } else if (item.agentType === 'literature_review') {
                agentLabel = 'Lit Review';
            } else if (item.agentType === 'deep_research') {
                agentLabel = 'Deep Research';
            }
            
            historyItem.innerHTML = `
                <div>
                    <div class="history-query">${item.query}</div>
                    <div class="history-date">${formattedDate}</div>
                </div>
                <div class="history-agent">${agentLabel}</div>
            `;
            
            // Add click event to repeat the search
            historyItem.addEventListener('click', () => {
                // Set values
                queryInput.value = item.query;
                
                // Select the agent type
                agentCards.forEach(card => {
                    if (card.getAttribute('data-agent') === item.agentType) {
                        card.click();
                    }
                });
                
                // Switch to search tab
                navItems[0].click();
                
                // Perform search
                searchButton.click();
            });
            
            historyList.appendChild(historyItem);
        });
    }
});
