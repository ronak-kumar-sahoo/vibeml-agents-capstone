document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const targetInput = document.getElementById("targetInput");
    const runBtn = document.getElementById("runBtn");
    const consoleWindow = document.getElementById("console");
    const progressBar = document.getElementById("progressBar");
    const resultsPanel = document.getElementById("resultsPanel");
    const reportContent = document.getElementById("reportContent");
    const chartsGallery = document.getElementById("chartsGallery");

    // State Variables
    let selectedFile = null;
    let pollInterval = null;
    let logIndex = 0;

    // Trigger file input click on dropzone click
    dropzone.addEventListener("click", () => fileInput.click());

    // File Drag and Drop Events
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Handle File Selection and Upload
    async function handleFileSelect(file) {
        if (!file.name.endsWith(".csv")) {
            addConsoleLine("Error: Only CSV files are allowed.", "error-msg");
            return;
        }

        selectedFile = file;
        dropzone.querySelector(".upload-text").innerHTML = `Selected: <span class="text-accent">${file.name}</span>`;
        dropzone.querySelector(".file-info").textContent = `Size: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
        addConsoleLine(`Selected dataset: ${file.name}`);

        // Upload to server
        addConsoleLine("Uploading dataset to server...");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error("File upload failed.");
            }

            const data = await response.json();
            addConsoleLine("Upload complete! Dataset is ready for analysis.", "success-msg");
            
            // Enable configurations
            targetInput.removeAttribute("disabled");
            runBtn.removeAttribute("disabled");
            targetInput.focus();

        } catch (error) {
            addConsoleLine(`Upload Error: ${error.message}`, "error-msg");
        }
    }

    // Run AutoML Pipeline
    runBtn.addEventListener("click", async () => {
        const targetColumn = targetInput.value.trim();
        if (!targetColumn) {
            addConsoleLine("Error: Target column name is required.", "error-msg");
            targetInput.focus();
            return;
        }

        // Lock UI controls
        targetInput.setAttribute("disabled", "true");
        runBtn.setAttribute("disabled", "true");
        resultsPanel.classList.add("hidden");
        
        // Reset indicators
        progressBar.style.width = "0%";
        document.querySelectorAll(".step-indicator").forEach(el => el.className = "step-indicator");
        consoleWindow.innerHTML = "";
        logIndex = 0;

        addConsoleLine(`Initiating AutoML loop. Target variable: '${targetColumn}'`, "system-msg");

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: selectedFile.name, target: targetColumn })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Failed to start pipeline.");
            }

            addConsoleLine("Agent orchestration thread spawned successfully.", "system-msg");
            
            // Start Polling Status
            pollInterval = setInterval(pollStatus, 1500);

        } catch (error) {
            addConsoleLine(`Pipeline Startup Error: ${error.message}`, "error-msg");
            targetInput.removeAttribute("disabled");
            runBtn.removeAttribute("disabled");
        }
    });

    // Poll logs and execution status
    async function pollStatus() {
        try {
            const response = await fetch("/api/status");
            const data = await response.json();

            // Print new logs
            if (data.logs && data.logs.length > logIndex) {
                for (let i = logIndex; i < data.logs.length; i++) {
                    const log = data.logs[i];
                    let logClass = "system-msg";

                    // Apply coloring based on agent prefix
                    if (log.includes("Profiler")) {
                        logClass = "profiler-msg";
                        updateStepUI("profiler", 30);
                    } else if (log.includes("ML Engineer")) {
                        logClass = "engineer-msg";
                        updateStepUI("engineer", 65);
                    } else if (log.includes("Reporter")) {
                        logClass = "reporter-msg";
                        updateStepUI("reporter", 90);
                    } else if (log.includes("Success!")) {
                        logClass = "success-msg";
                    } else if (log.includes("ERROR")) {
                        logClass = "error-msg";
                    }

                    addConsoleLine(log, logClass);
                }
                logIndex = data.logs.length;
            }

            // Check completion state
            if (data.status === "completed") {
                clearInterval(pollInterval);
                progressBar.style.width = "100%";
                document.querySelectorAll(".step-indicator").forEach(el => el.classList.add("completed"));
                addConsoleLine("Pipeline finished. Fetching evaluation assets...", "success-msg");
                
                await loadResults();

                // Unlock configuration UI
                targetInput.removeAttribute("disabled");
                runBtn.removeAttribute("disabled");

            } else if (data.status === "failed") {
                clearInterval(pollInterval);
                addConsoleLine("Pipeline Execution aborted due to system failures.", "error-msg");
                
                targetInput.removeAttribute("disabled");
                runBtn.removeAttribute("disabled");
            }

        } catch (error) {
            addConsoleLine(`Polling error: ${error.message}`, "error-msg");
        }
    }

    function updateStepUI(activeStep, progressWidth) {
        progressBar.style.width = `${progressWidth}%`;
        const profilerEl = document.getElementById("step-profiler");
        const engineerEl = document.getElementById("step-engineer");
        const reporterEl = document.getElementById("step-reporter");

        if (activeStep === "profiler") {
            profilerEl.className = "step-indicator active";
        } else if (activeStep === "engineer") {
            profilerEl.className = "step-indicator completed";
            engineerEl.className = "step-indicator active";
        } else if (activeStep === "reporter") {
            profilerEl.className = "step-indicator completed";
            engineerEl.className = "step-indicator completed";
            reporterEl.className = "step-indicator active";
        }
    }

    // Helper to print lines to the fake terminal
    function addConsoleLine(text, className = "system-msg") {
        const line = document.createElement("div");
        line.className = `console-line ${className}`;
        line.textContent = `> ${text}`;
        consoleWindow.appendChild(line);
        consoleWindow.scrollTop = consoleWindow.scrollHeight;
    }

    // Fetch and Render final AutoML Results
    async function loadResults() {
        try {
            const response = await fetch("/api/results");
            if (!response.ok) throw new Error("Could not retrieve model assets.");
            
            const results = await response.json();
            
            // Render Markdown Report
            reportContent.innerHTML = parseMarkdown(results.report);

            // Render Charts
            chartsGallery.innerHTML = "";
            if (results.plots && results.plots.length > 0) {
                results.plots.forEach(plotFile => {
                    const card = document.createElement("div");
                    card.className = "chart-card";

                    const titleText = plotFile.replace(".png", "").replace(/_/g, " ");
                    card.innerHTML = `
                        <h4>${titleText}</h4>
                        <div class="chart-img-wrapper">
                            <img src="/api/plots/${plotFile}?t=${Date.now()}" class="chart-img" alt="${titleText}">
                        </div>
                    `;
                    chartsGallery.appendChild(card);
                });
            } else {
                chartsGallery.innerHTML = "<p class='system-msg'>No plots were generated by the agent.</p>";
            }

            // Show results panel
            resultsPanel.classList.remove("hidden");
            resultsPanel.scrollIntoView({ behavior: "smooth" });

        } catch (error) {
            addConsoleLine(`Results Rendering Error: ${error.message}`, "error-msg");
        }
    }

    // A lightweight client-side Markdown to HTML converter
    function parseMarkdown(md) {
        if (!md) return "";
        let html = md;

        // Strip carriage returns
        html = html.replace(/\r\n/g, "\n");

        // Convert headers
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

        // Convert bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Convert bullet lists (simple parser)
        html = html.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
        // Wrap <li> sequences in <ul>
        // This is a simple regex that wraps adjacent <li> tags.
        html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');
        html = html.replace(/<\/ul>\s*<ul>/g, '\n');

        // Convert Code blocks
        html = html.replace(/```python([\s\S]*?)```/g, '<pre><code class="language-python">$1</code></pre>');
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

        // Convert simple markdown tables to HTML tables
        // Find tables and replace
        const tableRegex = /\|(.+)\|[\r\n]+\|([-:| ]+)\|[\r\n]+((?:\|.+\|[\r\n]*)+)/g;
        html = html.replace(tableRegex, (match, headerLine, alignLine, bodyLines) => {
            const headers = headerLine.split('|').map(h => h.trim()).filter(h => h);
            const headerHtml = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;

            const rows = bodyLines.trim().split('\n');
            const bodyHtml = rows.map(row => {
                const cols = row.split('|').map(c => c.trim()).filter(c => c);
                return `<tr>${cols.map(c => `<td>${c}</td>`).join('')}</tr>`;
            }).join('');

            return `<table><thead>${headerHtml}</thead><tbody>${bodyHtml}</tbody></table>`;
        });

        // Convert line breaks to paragraphs
        // Split by double newline, wrap non-HTML chunks in <p>
        const parts = html.split(/\n{2,}/);
        const wrappedParts = parts.map(part => {
            part = part.trim();
            if (!part) return "";
            if (part.startsWith("<h") || part.startsWith("<ul") || part.startsWith("<li") || part.startsWith("<pre") || part.startsWith("<table")) {
                return part;
            }
            return `<p>${part.replace(/\n/g, "<br>")}</p>`;
        });

        return wrappedParts.join("");
    }
});
