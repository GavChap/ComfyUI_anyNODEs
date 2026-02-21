import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "anyMODE.LoraXYGrid",
    async nodeCreated(node) {
        if (node.comfyClass === "LoraXYIntegratedSampler" || node.comfyClass === "LoraXYIntegratedSamplerCustom") {
            setTimeout(() => {
                // Add the copy button
                node.addWidget("button", "Copy LoRA 1 to All", null, () => {
                    const lora1Widget = node.widgets.find(w => w.name === "lora_1");
                    if (!lora1Widget) {
                        console.error("anyMODE: Could not find lora_1 widget");
                        return;
                    }

                    const val = lora1Widget.value;
                    for (let i = 2; i <= 10; i++) {
                        const w = node.widgets.find(widget => widget.name === `lora_${i}`);
                        if (w) {
                            w.value = val;
                        }
                    }

                    // Trigger canvas refresh
                    if (node.graph) {
                        node.setDirtyCanvas(true, true);
                    }
                });

                // Add the clear button
                node.addWidget("button", "Clear All LoRAs", null, () => {
                    for (let i = 1; i <= 10; i++) {
                        const w = node.widgets.find(widget => widget.name === `lora_${i}`);
                        if (w) {
                            w.value = "None";
                        }
                    }

                    // Trigger canvas refresh
                    if (node.graph) {
                        node.setDirtyCanvas(true, true);
                    }
                });
            }, 100);
        }
    }
});
