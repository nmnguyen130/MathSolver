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

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (data.equation) {
      latexOutput.value = data.equation;
      renderMath(data.equation);
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
    // Simulate API call to detect formula
    const formData = new FormData();
    formData.append("equation", latex);
    let query;
    if (latex.includes("x")) {
      query = "Tìm x";
    } else {
      query = "Tính";
    }
    formData.append("query", query);

    const response = await fetch("http://127.0.0.1:5000/solve", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    console.log(data);
    if (data.solution) {
      // Display final solution
      renderMath(data.solution);

      // Display solution steps if available
      if (data.steps) {
        const stepsContent = document.getElementById("stepsContent");
        stepsContent.innerHTML = "";
        data.steps.forEach((step, index) => {
          const stepDiv = document.createElement("div");
          stepDiv.className = "step";
          stepDiv.innerHTML = `
            <div class="step-number">Step ${index + 1}:</div>
            <div class="step-content">\(${step}\)</div>
          `;
          stepsContent.appendChild(stepDiv);
        });
        MathJax.typesetPromise([stepsContent]).catch((err) => {
          console.error("MathJax error:", err);
        });
        document.getElementById("solutionSteps").style.display = "block";
      }
    } else {
      mathOutput.innerHTML = "No solution found";
    }
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
