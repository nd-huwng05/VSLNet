(function (window) {
    class PoseHeader {
        constructor(data) {
            this.version = 0.1;
            this.width = data.dimensions?.width || 640;
            this.height = data.dimensions?.height || 480;
            this.depth = data.dimensions?.depth || 1000;
            this.components = data.components || [];
        }
    }

    class PoseBody {
        constructor(fps, data, confidence) {
            this.fps = fps;
            this.data = data;
            this.confidence = confidence;
        }
    }

    class DTPose {
        constructor(header, body) {
            this.header = header;
            this.body = body;
        }

        async write() {
            const encoder = new TextEncoder();
            const chunks = [];

            const vBuf = new ArrayBuffer(4);
            new DataView(vBuf).setFloat32(0, this.header.version, true);
            chunks.push(new Uint8Array(vBuf));

            const dimBuf = new ArrayBuffer(6);
            const dimView = new DataView(dimBuf);
            dimView.setUint16(0, this.header.width, true);
            dimView.setUint16(2, this.header.height, true);
            dimView.setUint16(4, this.header.depth, true);
            chunks.push(new Uint8Array(dimBuf));

            const compCountBuf = new ArrayBuffer(2);
            new DataView(compCountBuf).setUint16(0, this.header.components.length, true);
            chunks.push(new Uint8Array(compCountBuf));

            for (const comp of this.header.components) {
                const writeStr = (str) => {
                    const bytes = encoder.encode(str);
                    const lenBuf = new ArrayBuffer(2);
                    new DataView(lenBuf).setUint16(0, bytes.length, true);
                    chunks.push(new Uint8Array(lenBuf));
                    chunks.push(bytes);
                };

                writeStr(comp.name);
                writeStr(comp.format);

                const pCountBuf = new ArrayBuffer(2);
                new DataView(pCountBuf).setUint16(0, comp.points.length, true);
                chunks.push(new Uint8Array(pCountBuf));

                for (const p of comp.points) {
                    writeStr(p);
                }

                const lcBuf = new ArrayBuffer(4);
                const lcView = new DataView(lcBuf);
                lcView.setUint16(0, 0, true);
                lcView.setUint16(2, 0, true);
                chunks.push(new Uint8Array(lcBuf));
            }

            const numFrames = this.body.data.length / (75 * 3);
            const bodyHeadBuf = new ArrayBuffer(4);
            const bhView = new DataView(bodyHeadBuf);
            bhView.setUint16(0, this.body.fps, true);
            bhView.setUint16(2, numFrames, true);
            chunks.push(new Uint8Array(bodyHeadBuf));

            chunks.push(new Uint8Array(this.body.data.buffer, this.body.data.byteOffset, this.body.data.byteLength));
            chunks.push(new Uint8Array(this.body.confidence.buffer, this.body.confidence.byteOffset, this.body.confidence.byteLength));

            const totalLen = chunks.reduce((acc, c) => acc + c.length, 0);
            const finalArray = new Uint8Array(totalLen);
            let offset = 0;
            for (const chunk of chunks) {
                finalArray.set(chunk, offset);
                offset += chunk.length;
            }

            return finalArray.buffer;
        }
    }

    window.PoseHeader = PoseHeader;
    window.PoseBody = PoseBody;
    window.DTPose = DTPose;
})(window);