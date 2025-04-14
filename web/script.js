// Image Upload Component Logic
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const detectBtn = document.getElementById("detectBtn");
let uploadedImage = null;

imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      preview.src = event.target.result;
      preview.style.display = "block";
      uploadedImage = file;
      detectBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }
});

// LaTeX Output Component Logic
const latexOutput = document.getElementById("latexOutput");
const clearBtn = document.getElementById("clearBtn");
const solveBtn = document.getElementById("solveBtn");
const copyBtn = document.getElementById("copyBtn");

detectBtn.addEventListener("click", async () => {
  if (!uploadedImage) return;
  detectBtn.disabled = true;
  try {
    // Simulate API call to detect formula
    const formData = new FormData();
    formData.append("image", uploadedImage);

    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (data.latex) {
      latexOutput.value = data.latex;
      renderMath(data.latex);
      solveBtn.disabled = false;
      copyBtn.disabled = false;
    } else {
      latexOutput.value = "Error: Error detecting formula.";
    }
  } catch (error) {
    latexOutput.value = "Error: Error conecting to server.";
  } finally {
    detectBtn.disabled = false;
  }
});

clearBtn.addEventListener("click", () => {
  imageInput.value = "";
  preview.style.display = "none";
  latexOutput.value = "";
  document.getElementById("mathOutput").innerHTML = "";
  detectBtn.disabled = true;
  solveBtn.disabled = true;
  copyBtn.disabled = true;
  uploadedImage = null;
});

// Result Component Logic
const mathOutput = document.getElementById("mathOutput");

function renderMath(latex) {
  // Display rendered formula instead of LaTeX code
  mathOutput.innerHTML = `\\(${latex}\\)`;
  MathJax.typesetPromise([mathOutput]).catch((err) => {
    console.error("MathJax error:", err);
    mathOutput.innerHTML = "Error rendering formula";
  });
}

solveBtn.addEventListener("click", async () => {
  const latex = latexOutput.value;
  if (!latex) return;
  solveBtn.disabled = true;
  try {
    // Simulate solving (replace with actual solver API call)
    // const response = await fetch('YOUR_SOLVER_API_ENDPOINT', {
    //     method: 'POST',
    //     body: JSON.stringify({ latex })
    // });
    // const data = await response.json();

    // Simulated solver response
    const solution = "\\frac{(x + 1)^2}{x - 1}";
    renderMath(solution); // Render solution as formula, not LaTeX code
  } catch (error) {
    mathOutput.innerHTML = "Error solving formula";
  } finally {
    solveBtn.disabled = false;
  }
});

copyBtn.addEventListener("click", () => {
  const latex = latexOutput.value;
  if (latex) {
    navigator.clipboard
      .writeText(latex)
      .then(() => alert("LaTeX copied to clipboard!"))
      .catch(() => alert("Failed to copy LaTeX"));
  }
});
