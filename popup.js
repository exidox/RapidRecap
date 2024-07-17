const sumbtn = document.getElementById("summarize");
const p = document.getElementById("summary");
const clrbtn = document.getElementById("clear")

chrome.storage.local.get(['summary'], function(result) {
    if (result.summary) {
        p.innerHTML = result.summary;
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
            
            p.innerHTML = text;

            chrome.storage.local.set({summary: text}, function() {
                sumbtn.disabled = false;
                sumbtn.innerHTML = "Summarize";
            });
            
        };

        /* xmlreq.onerror = function() {
            const p = document.getElementById("summary");
            p.innerHTML = "An error occurred while fetching the summary.";
            sumbtn.disabled = false;
            sumbtn.innerHTML = "Summarize";
        }; */
        xmlreq.send();

    });
});

clrbtn.addEventListener("click", function(){
    p.innerHTML = "";
    chrome.storage.local.remove('summary');
});




