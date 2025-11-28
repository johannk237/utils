import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';




// --- Icons (Using actual SVGs now) ---
const UploadIcon = ({ className = "" }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 inline-block ${className}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
  </svg>
);

const TextIcon = ({ className = "" }: { className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 inline-block ${className}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
    </svg>
);

const CheckCircleIcon = ({ className = "" }: { className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 inline-block ${className}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);

const ExclamationCircleIcon = ({ className = "" }: { className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 inline-block ${className}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);

const LoadingSpinner = () => (
    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
);
// --- End Icons ---

// --- Configuration ---
const API_BASE_URL = 'http://localhost:8000'; // Centralized API URL
const POLLING_INTERVAL_MS = 3000; // Interval for status checks
// --- End Configuration ---


// Define the expected API response structure
// Initial response from POST /process-names/ or /process-names-text/
interface InitialApiResponse {
  message: string;
  task_id: string; // Expecting the backend to return a task ID
  // These might be included in the initial response for prediction, but are confirmed in status
  // input_filename?: string;
  // output_file?: string;
  // log_file?: string;
  detail?: string; // For FastAPI HTTPExceptions
}

// Response from GET /task-status/{task_id}
// Matches the structure stored in task_statuses in main_api.py
interface TaskStatusResponse {
  task_id: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  progress_message: string; // e.g., "Searching Google for 'Jane Doe' (2/10)"
  input_filename?: string; // Added based on backend structure
  output_file?: string; // Final output file path when completed
  log_file?: string;    // Final log file path when completed
  error_detail?: string; // Details if status is FAILED
}


type InputMode = 'file' | 'text';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [textInput, setTextInput] = useState<string>('');
  const [inputMode, setInputMode] = useState<InputMode>('file');
  const [statusMessage, setStatusMessage] = useState<string>('');
  // isLoading now primarily tracks the initial submission request,
  // while taskId indicates an active background process being polled.
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isError, setIsError] = useState<boolean>(false);
  const [taskId, setTaskId] = useState<string | null>(null);

  // Ref to store the polling interval ID
  const pollingIntervalRef = useRef<number | null>(null);

  // --- Input Handlers ---
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    // Clear previous task/status if user interacts with input
    stopPolling();
    setTaskId(null);
    setIsLoading(false); // Ensure loading is false if user changes input

    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        setSelectedFile(file);
        setTextInput(''); // Clear text input
        setStatusMessage(`Selected file: ${file.name}`);
        setIsError(false);
      } else {
        setSelectedFile(null);
        setStatusMessage('Error: Please select a valid CSV file.');
        setIsError(true);
        event.target.value = ''; // Reset file input visually on error
      }
    } else {
      // User cancelled file selection
      setSelectedFile(null);
      // Clear status only if it wasn't an error message
      if (!isError) setStatusMessage('');
    }
  };

  const handleTextChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newText = event.target.value;
    setTextInput(newText);

    // Clear previous task/status if user interacts with input
    stopPolling();
    setTaskId(null);
    setIsLoading(false); // Ensure loading is false if user changes input

    // Clear file input if text is entered
    if (newText) {
        setSelectedFile(null);
        const fileInput = document.getElementById('csv-upload') as HTMLInputElement;
        if (fileInput) fileInput.value = ''; // Reset file input visually
    }
    // Clear status/error if user starts typing valid input
    if (newText.trim()) {
        setStatusMessage('');
        setIsError(false);
    }
  };

  const handleModeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputMode(event.target.value as InputMode);
    // Clear everything related to the previous mode and any running task
    setSelectedFile(null);
    setTextInput('');
    setStatusMessage('');
    setIsError(false);
    setTaskId(null);
    setIsLoading(false);
    stopPolling();
    // Reset file input visually if it exists
    const fileInput = document.getElementById('csv-upload') as HTMLInputElement;
    if (fileInput) fileInput.value = '';
  };
  // --- End Input Handlers ---


  const canSubmit = useMemo(() => {
    // Prevent submission if a request is actively being made (isLoading)
    // or if a background task is being polled (taskId is set).
    if (isLoading || taskId) return false;
    // Check for valid input based on the selected mode.
    if (inputMode === 'file') return selectedFile !== null;
    if (inputMode === 'text') return textInput.trim() !== '';
    return false; // Should not happen in practice
  }, [isLoading, taskId, inputMode, selectedFile, textInput]);


  // --- Polling Logic ---
  const stopPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
      console.log("Polling stopped.");
    }
  }, []); // No dependencies needed, clearInterval is stable

  const fetchTaskStatus = useCallback(async (currentTaskId: string) => {
    console.log(`Polling status for task: ${currentTaskId}`);
    const statusUrl = `${API_BASE_URL}/task-status/${currentTaskId}`;

    try {
      const response = await fetch(statusUrl);

      // Handle Task Not Found specifically
      if (response.status === 404) {
        throw new Error(`Task ${currentTaskId} not found on the server. It might have expired or never existed.`);
      }

      // Handle other non-OK HTTP statuses
      if (!response.ok) {
        // Try to get error detail from response body, fallback to status text
        let errorDetail = `Server error: ${response.status} ${response.statusText}`;
        try {
            const errorResult = await response.json();
            if (errorResult.detail) {
                errorDetail = errorResult.detail;
            }
        } catch (jsonError) {
            // Ignore if response is not JSON or empty
        }
        throw new Error(errorDetail);
      }

      // Process successful response
      const result = (await response.json()) as TaskStatusResponse;

      setStatusMessage(result.progress_message || 'Processing...'); // Update with the detailed message
      setIsError(result.status === 'FAILED');

      // Stop polling and reset state if task is completed or failed
      if (result.status === 'COMPLETED' || result.status === 'FAILED') {
        stopPolling();
        setIsLoading(false); // Task is no longer actively running/submitting
        setTaskId(null); // Clear task ID as it's finished

        // Construct final status message
        let finalMessage = result.progress_message || (result.status === 'COMPLETED' ? 'Task completed.' : 'Task failed.');
        if (result.status === 'COMPLETED') {
            if (result.output_file) finalMessage += ` Output: ${result.output_file}.`;
            if (result.log_file) finalMessage += ` Log: ${result.log_file}.`;
        } else if (result.status === 'FAILED') {
            // Prepend "Error:" for clarity and include detail if available
            finalMessage = `Error: ${result.error_detail || finalMessage}`;
        }
        setStatusMessage(finalMessage);
      }
      // If status is PENDING or PROCESSING, polling continues automatically via setInterval

    } catch (error) {
      // Handle network errors or errors thrown above
      console.error('Polling failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred while checking task status.';
      setStatusMessage(`Error: ${errorMessage}`);
      setIsError(true);
      setIsLoading(false); // Stop loading indicator on polling failure
      stopPolling();
      setTaskId(null); // Clear task ID on polling error
    }
  }, [stopPolling]); // Include stopPolling in dependencies


  const startPolling = useCallback((newTaskId: string) => {
    stopPolling(); // Ensure no previous interval is running
    console.log(`Starting polling for task: ${newTaskId}`);

    // Set the task ID in state *before* the first fetch
    setTaskId(newTaskId);
    // Note: isLoading remains true from the handleSubmit function initially

    // Poll immediately to get the first status update quickly
    fetchTaskStatus(newTaskId);

    // Set up the interval for subsequent polls
    pollingIntervalRef.current = setInterval(() => {
      // Pass the taskId from the state at the time the interval runs
      // This ensures it uses the correct ID even if things change rapidly (though unlikely here)
      setTaskId(prevTaskId => {
        if (prevTaskId) {
          fetchTaskStatus(prevTaskId);
        } else {
          // Should not happen if interval is cleared correctly, but as a safeguard:
          stopPolling();
        }
        return prevTaskId; // Keep the state the same within this setter
      });
    }, POLLING_INTERVAL_MS);

  }, [fetchTaskStatus, stopPolling]); // Include dependencies

  // Cleanup polling interval on component unmount
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]); // Dependency array is correct
  // --- End Polling Logic ---


  const handleSubmit = useCallback(async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    // Double-check if submission is allowed (belt and suspenders)
    if (!canSubmit) {
        setStatusMessage('Error: Cannot submit. Please provide input or wait for the current process to finish.');
        setIsError(true);
        return;
    }

    // Reset state for the new submission
    setIsLoading(true); // Indicate the initial submission process is starting
    setIsError(false);
    setStatusMessage(inputMode === 'file' && selectedFile ? `Uploading ${selectedFile.name}...` : 'Initiating processing...');
    setTaskId(null); // Ensure no old task ID is present
    stopPolling(); // Ensure no previous polling is running

    let requestUrl: string;
    let requestOptions: RequestInit;

    // --- Prepare request based on input mode ---
    if (inputMode === 'file' && selectedFile) {
        requestUrl = `${API_BASE_URL}/process-names/`;
        const formData = new FormData();
        formData.append('file', selectedFile);
        requestOptions = { method: 'POST', body: formData };
    } else if (inputMode === 'text') {
        requestUrl = `${API_BASE_URL}/process-names-text/`;
        const namesArray = textInput.split('\n').map(name => name.trim()).filter(name => name !== '');
        requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ names: namesArray }),
        };
    } else {
        // This case should be prevented by `canSubmit`, but handle defensively
        setStatusMessage('Error: Invalid input mode selected.');
        setIsError(true);
        setIsLoading(false); // Reset loading state
        return;
    }
    // --- End request preparation ---

    try {
      // --- Initial POST request to start the task ---
      const response = await fetch(requestUrl, requestOptions);
      const result = (await response.json()) as InitialApiResponse | { detail: string }; // Handle potential FastAPI error response

      if (!response.ok) {
        // Handle errors from the initial request (e.g., validation errors from FastAPI)
        const errorDetail = (result as { detail: string }).detail || `Server error: ${response.status}`;
        throw new Error(errorDetail);
      }

      // --- Task started successfully, begin polling ---
      // Type guard to ensure result has task_id
      if ('task_id' in result && result.task_id) {
        setStatusMessage(result.message || "Processing started, checking status..."); // Update confirmation message
        startPolling(result.task_id);
        // setIsLoading(true) remains true here, indicating the overall process (including polling) is active.
        // The button text will show "Processing..."
      } else {
        // Backend didn't return a task_id as expected
        throw new Error("Backend did not return a task ID to track progress.");
      }

    } catch (error) {
      // Handle network errors or errors thrown from the try block
      console.error('Submission failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to start processing.';
      setStatusMessage(`Error: ${errorMessage}`);
      setIsError(true);
      setIsLoading(false); // Stop loading indicator on submission failure
      stopPolling(); // Ensure polling is stopped if it somehow started
      setTaskId(null); // Clear any potential task ID
    }
    // Note: `isLoading` is intentionally NOT set to false here if polling starts.
    // It will be set to false by `fetchTaskStatus` when the task completes or fails.
  }, [selectedFile, textInput, inputMode, canSubmit, startPolling, stopPolling]); // Dependencies are correct


  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100 p-4 font-sans antialiased">
      <div className="bg-white p-6 sm:p-8 rounded-xl shadow-xl w-full max-w-xl border border-slate-200">
        {/* --- Header --- */}
        <h1 className="text-2xl sm:text-3xl font-bold text-center text-slate-800 mb-8">
          Contact & Profile Finder
        </h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* --- Input Mode Selection (Disable when loading or polling) --- */}
           <fieldset className="border border-slate-200 rounded-lg p-4">
             <legend className="text-sm font-medium text-slate-600 px-2">Input Method</legend>
             <div className="flex flex-col sm:flex-row sm:items-center sm:space-x-6 space-y-3 sm:space-y-0">
               {/* File Input Radio */}
               <label className={`flex items-center space-x-2 p-2 rounded-md transition-colors duration-150 ${isLoading || taskId ? 'cursor-not-allowed opacity-60' : 'cursor-pointer hover:bg-indigo-50'}`}>
                 <input
                   type="radio"
                   name="inputMode"
                   value="file"
                   checked={inputMode === 'file'}
                   onChange={handleModeChange}
                   disabled={isLoading || !!taskId} // Disable if submitting or polling
                   className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-slate-300 disabled:opacity-75"
                 />
                 <span className="text-slate-700 font-medium">Upload CSV File</span>
               </label>
               {/* Text Input Radio */}
               <label className={`flex items-center space-x-2 p-2 rounded-md transition-colors duration-150 ${isLoading || taskId ? 'cursor-not-allowed opacity-60' : 'cursor-pointer hover:bg-indigo-50'}`}>
                 <input
                   type="radio"
                   name="inputMode"
                   value="text"
                   checked={inputMode === 'text'}
                   onChange={handleModeChange}
                   disabled={isLoading || !!taskId} // Disable if submitting or polling
                   className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-slate-300 disabled:opacity-75"
                 />
                  <span className="text-slate-700 font-medium">Enter Names Manually</span>
               </label>
             </div>
           </fieldset>

          {/* --- Conditional Input Area (Disable when loading or polling) --- */}
          <div className="border border-slate-200 rounded-lg p-4 bg-slate-50/50">
            {inputMode === 'file' ? (
              // File Input Area
              <div>
                <label
                  htmlFor="csv-upload"
                  className={`block text-sm font-medium mb-2 ${isLoading || taskId ? 'text-slate-500' : 'text-slate-700'}`}
                >
                  <UploadIcon className="text-gray-500 mr-1" /> Select CSV File <span className="text-slate-500">(must contain 'nom' column)</span>
                </label>
                <input
                  id="csv-upload"
                  type="file"
                  accept=".csv, text/csv"
                  onChange={handleFileChange}
                  disabled={isLoading || !!taskId} // Disable if submitting or polling
                  className={`block w-full text-sm rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500
                    file:mr-4 file:py-2 file:px-4 file:rounded-l-md file:border-0
                    file:text-sm file:font-semibold
                    file:bg-indigo-100 file:text-indigo-700
                    ${!(isLoading || taskId) ? 'hover:file:bg-indigo-200 cursor-pointer' : ''}
                    transition-colors duration-150
                    disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:file:bg-slate-200
                    ${isError && inputMode === 'file' && !taskId ? 'border-red-500 ring-1 ring-red-500' : 'text-slate-600'}
                  `} // Show error ring only if not polling
                />
                 {/* Show selected file name only if valid and not processing */}
                 {selectedFile && !isError && !(isLoading || taskId) && (
                    <p className="mt-2 text-xs text-green-700">Selected: {selectedFile.name}</p>
                 )}
              </div>
            ) : (
              // Text Input Area
              <div>
                <label
                  htmlFor="text-input"
                  className={`block text-sm font-medium mb-2 ${isLoading || taskId ? 'text-slate-500' : 'text-slate-700'}`}
                >
                  <TextIcon className="text-gray-500 mr-1" /> Enter Names <span className="text-slate-500">(one per line)</span>
                </label>
                <textarea
                  id="text-input"
                  rows={6}
                  value={textInput}
                  onChange={handleTextChange}
                  disabled={isLoading || !!taskId} // Disable if submitting or polling
                  placeholder="Example:&#10;John Doe&#10;Jane Smith&#10;Acme Corporation CEO"
                  className={`block w-full p-3 text-sm text-slate-900 bg-white rounded-lg border border-slate-300 focus:ring-indigo-500 focus:border-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-slate-100 transition-colors duration-150 ${isError && inputMode === 'text' && !taskId ? 'border-red-500 ring-1 ring-red-500' : ''}`} // Show error ring only if not polling
                />
              </div>
            )}
          </div>

          {/* --- Submit Button --- */}
          <button
            type="submit"
            // Disable using the memoized 'canSubmit' which checks isLoading, taskId, and input validity
            disabled={!canSubmit}
            className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed disabled:bg-indigo-400 transition-all duration-150 ease-in-out"
          >
            {/* Show spinner/text based on isLoading (initial submit) or taskId (polling) */}
            {isLoading || taskId ? (
              <>
                <LoadingSpinner />
                Processing...
              </>
            ) : (
              'Start Processing'
            )}
          </button>
        </form>

        {/* --- Status Message Area --- */}
        {/* Show status message if it exists */}
        {statusMessage && (
          <div className={`mt-6 p-4 rounded-lg text-sm border flex items-start ${
              isError
                ? 'bg-red-50 text-red-800 border-red-300'
                : 'bg-green-50 text-green-800 border-green-300'
            }`}
            role="alert"
          >
            {/* Show appropriate icon based on error state */}
            {isError ? <ExclamationCircleIcon className="flex-shrink-0 mr-2 mt-0.5 text-red-600" /> : <CheckCircleIcon className="flex-shrink-0 mr-2 mt-0.5 text-green-600" />}
            {/* Display the status message */}
            <span className="flex-grow">{statusMessage}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
