import { useState } from "react";
import axios from "axios";

function App() {
  const [caseName, setCaseName] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const createCase = async () => {
    const res = await axios.post("http://127.0.0.1:8000/cases/create", {
      case_name: caseName,
    });
    alert(res.data.message);
  };

  const askQuestion = async () => {
    const res = await axios.post("http://127.0.0.1:8000/ask", {
      case_name: caseName,
      query: question,
    });
    setAnswer(res.data.answer);
  };

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold">⚖️ Legal AI Assistant</h1>

      <div className="mt-4">
        <input
          value={caseName}
          onChange={(e) => setCaseName(e.target.value)}
          placeholder="Enter case name"
          className="border p-2"
        />
        <button onClick={createCase} className="ml-2 bg-blue-500 text-white px-4 py-2">
          Create Case
        </button>
      </div>

      <div className="mt-4">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a legal question"
          className="border p-2 w-96"
        />
        <button onClick={askQuestion} className="ml-2 bg-green-500 text-white px-4 py-2">
          Ask
        </button>
      </div>

      {answer && (
        <div className="mt-4 p-4 border rounded bg-gray-100">
          <h2 className="font-semibold">💡 Answer:</h2>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;
