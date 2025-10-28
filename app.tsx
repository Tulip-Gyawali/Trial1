import logo from "./assets/youware-bg.png";

function App() {
  return (
    <main 
      className="min-h-screen flex items-center justify-center relative bg-[#F6F4F1] bg-cover bg-center bg-no-repeat"
      style={{ 
        backgroundImage: `url(${logo})`,
      }}
    >
      <div className="text-center z-10 max-w-4xl mx-auto px-6 py-8">
        <h1 className="text-black font-normal leading-tight text-center mb-4 text-4xl sm:text-5xl lg:text-6xl">
          YouWare App Template
        </h1>
        
        <p className="text-black font-normal leading-relaxed text-center max-w-2xl mx-auto text-base sm:text-lg lg:text-xl">
          Start building your React application here
        </p>
      </div>
    </main>
  );
}

export default App;
