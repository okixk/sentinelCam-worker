Update the HTML web server and frontend so it uses the worker's new WebRTC stream instead of MJPEG.

Context:
- The worker now supports WebRTC signaling at `POST /api/webrtc/offer`.
- The worker still exposes `GET /api/state`, `POST /api/cmd`, and `GET /health`.
- The worker may be protected with `WEB_AUTH_TOKEN`, so the browser should not talk to the worker directly in production.
- The preferred architecture is: browser -> HTML web server / reverse proxy -> worker.

Required changes:

1. Replace MJPEG rendering with WebRTC video
- Find any `<img>` or similar element that currently renders `/stream.mjpg`, `/frame.jpg`, or other MJPEG endpoints.
- Replace it with a `<video>` element:
  - `autoplay`
  - `playsinline`
  - `muted`
- Example target:
  - `<video id="stream" autoplay playsinline muted></video>`

2. Add browser-side WebRTC startup logic
- Create an `RTCPeerConnection`.
- Add one video transceiver with `direction: "recvonly"`.
- Listen for `pc.ontrack` and attach the first received stream to the `<video>` element via `video.srcObject = stream`.
- Create an SDP offer with `pc.createOffer()`.
- Call `pc.setLocalDescription(offer)`.
- Wait until ICE gathering is complete before sending the offer.
- Send the offer JSON to `/api/webrtc/offer` with:
  - method `POST`
  - header `Content-Type: application/json`
  - body:
    - `{ "sdp": pc.localDescription.sdp, "type": pc.localDescription.type }`
- Parse the JSON response and call `pc.setRemoteDescription(answer)`.
- Add basic reconnect / retry handling if negotiation fails or the connection state becomes `failed`, `disconnected`, or `closed`.

3. Keep the existing control/state API integration
- Keep using `GET /api/state` for UI state polling.
- Keep using `POST /api/cmd` for actions like next model, previous model, toggle pose, overlay, inference, stop, etc.
- Do not remove existing state/control UI unless it depends directly on MJPEG behavior.

4. Add cleanup logic
- When the page is hidden, unloaded, or the component is destroyed, close the `RTCPeerConnection`.
- Avoid creating multiple active peer connections for the same viewer unless explicitly intended.

5. Reverse proxy the worker endpoints through the HTML web server
- Proxy these routes from the HTML server to the worker:
  - `/api/webrtc/offer`
  - `/api/state`
  - `/api/cmd`
  - `/health`
- If the worker uses token auth, the reverse proxy must add one of these headers on proxied requests:
  - `Authorization: Bearer <WORKER_TOKEN>`
  - or `X-SentinelCam-Token: <WORKER_TOKEN>`
- Do not expose the worker token in browser JavaScript for production.
- Do not place the token in query params.

6. Security requirements
- Serve the HTML app over HTTPS in production, or only use `http://localhost` for local development.
- Keep the worker bound to `127.0.0.1` when possible.
- If cross-origin browser access is used, ensure the worker has an explicit `WEB_ALLOWED_ORIGINS` allowlist configured.
- Do not use wildcard origins.

7. Fallback behavior
- If WebRTC setup fails, show a visible error message in the UI.
- Optionally add a fallback mode that uses MJPEG only when WebRTC is unavailable.
- If a fallback exists, make it explicit and easy to diagnose in the UI.

8. Deliverables
- Updated HTML markup.
- Updated frontend JavaScript or framework component code.
- Any required backend/reverse-proxy changes.
- A short note describing where the worker token is configured server-side.
- A short note describing how to switch between WebRTC and MJPEG if both are supported.

Implementation reference:
- Signaling endpoint: `POST /api/webrtc/offer`
- State endpoint: `GET /api/state`
- Command endpoint: `POST /api/cmd`
- Health endpoint: `GET /health`

Use this browser logic pattern:

```html
<video id="stream" autoplay playsinline muted></video>
<script>
  async function startWebRtcStream() {
    const video = document.getElementById("stream");
    const pc = new RTCPeerConnection();

    pc.addTransceiver("video", { direction: "recvonly" });

    pc.ontrack = (event) => {
      const [stream] = event.streams;
      if (stream) {
        video.srcObject = stream;
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    await new Promise((resolve) => {
      if (pc.iceGatheringState === "complete") {
        resolve();
        return;
      }
      const onStateChange = () => {
        if (pc.iceGatheringState === "complete") {
          pc.removeEventListener("icegatheringstatechange", onStateChange);
          resolve();
        }
      };
      pc.addEventListener("icegatheringstatechange", onStateChange);
    });

    const response = await fetch("/api/webrtc/offer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type
      })
    });

    if (!response.ok) {
      throw new Error(`WebRTC offer failed: ${response.status}`);
    }

    const answer = await response.json();
    await pc.setRemoteDescription(answer);
    return pc;
  }
</script>
```

Output expectation:
- Modify the existing HTML/web app so the camera view uses WebRTC by default.
- Keep controls working through `/api/state` and `/api/cmd`.
- Route signaling through `/api/webrtc/offer`.
- Keep the implementation production-safe regarding token handling.
