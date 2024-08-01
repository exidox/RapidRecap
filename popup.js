const sumbtn = document.getElementById("summarize");
const p1 = document.getElementById("summary");

const mcqbtn = document.getElementById("gen_mcq");
const p2 = document.getElementById("mcq");


const clrbtn = document.getElementById("clear")

chrome.storage.local.get(['summary'], function(result) {
    if (result.summary) {
        p1.innerHTML = result.summary;
    }
});

chrome.storage.local.get(['mcq'], function(result) {
    if (result.mcq) {
        /* p2.innerHTML = result.mcq; */
        displayMcqs(result.mcq)
    }
});

sumbtn.addEventListener("click", function(){
    sumbtn.disabled = true;
    sumbtn.innerHTML = "Summarizing ...";
    chrome.tabs.query({currentWindow: true, active: true}, function(tabs){
        var url = tabs[0].url;
        var xmlreq = new XMLHttpRequest();
        xmlreq.open("GET", "http://127.0.0.1:5000/summary?url=" + encodeURIComponent(url), true);
        xmlreq.onload = function(){
            var text = xmlreq.responseText;
            
            p1.innerHTML = text;

            chrome.storage.local.set({summary: text}, function() {
                sumbtn.disabled = false;
                sumbtn.innerHTML = "summarize";
            });  
        };
        xmlreq.onerror = function() {
            const p = document.getElementById("summary");
            p.innerHTML = "An error occurred while fetching the summary.";
            sumbtn.disabled = false;
            sumbtn.innerHTML = "Summarize";
        };
        xmlreq.send();

    });
});


mcqbtn.addEventListener("click", function(){
    mcqbtn.disabled = true;
    mcqbtn.innerHTML = "MCQing ...";
    chrome.tabs.query({currentWindow: true, active: true}, function(tabs){
        var url = tabs[0].url;
        var xmlreq = new XMLHttpRequest();
        xmlreq.open("GET", "http://127.0.0.1:5000/mcq?url=" + encodeURIComponent(url), true);
        xmlreq.onload = function(){
            var response = xmlreq.responseText;
            try {
                var data = JSON.parse(response);
                if (data.error) {
                    p2.innerHTML = "Error: " + data.error;
                } else {

                    displayMcqs(data);

                    /* p2.innerHTML = JSON.stringify(data, null, 2); */
                }
            } catch (e) {
                p2.innerHTML = "Failed to parse response as JSON.";
            }

            chrome.storage.local.set({mcq: data}, function() {
                mcqbtn.disabled = false;
                mcqbtn.innerHTML = "mcq test";
            });
        };
        xmlreq.onerror = function() {
            p2.innerHTML = "Request failed.";
            mcqbtn.disabled = false;
            mcqbtn.innerHTML = "mcq test";
        };
        xmlreq.send();
    });
});

function displayMcqs(mcqData) {
    const mcqContainer = document.getElementById('mcq');
    mcqContainer.innerHTML = '';

    mcqData.forEach((questionArray, index) => {
        const questionDiv = document.createElement('div');
        questionDiv.className = 'mcq-question';

        questionArray.forEach((text, i) => {
            const p = document.createElement('p');
            p.textContent = text;
            questionDiv.appendChild(p);
        });

        mcqContainer.appendChild(questionDiv);
    });
}




clrbtn.addEventListener("click", function(){
    p2.innerHTML = "";
    chrome.storage.local.remove('mcq');

    p1.innerHTML= "";
    chrome.storage.local.remove('summary')
});






