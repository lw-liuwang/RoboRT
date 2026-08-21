// RoboRT vla-pi05-server 最小验证客户端（aarch64 交叉编译，Orin 本地自测用）
//
// 用法: ./vla-pi05-client [tcp://127.0.0.1:5555]
// 发一个最小 PredictRequest（lang_tokens + state，无图像/noise），
// 打印 PredictResponse 的 request_id / chunk_size / action_dim / 延迟 与前 8 个 action。
#include <zmq.h>
#include <cstdio>
#include <cstring>
#include <string>
#include "vla.pb.h"

int main(int argc, char **argv) {
    const char *addr = argc > 1 ? argv[1] : "tcp://127.0.0.1:5555";

    void *ctx = zmq_ctx_new();
    if (!ctx) { fprintf(stderr, "zmq_ctx_new failed\n"); return 1; }
    void *sock = zmq_socket(ctx, ZMQ_REQ);
    if (!sock) { fprintf(stderr, "zmq_socket failed\n"); return 1; }
    int rcvtimeo = 120000;  // ms
    zmq_setsockopt(sock, ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
    if (zmq_connect(sock, addr) != 0) {
        fprintf(stderr, "zmq_connect(%s) failed\n", addr);
        return 1;
    }

    vla::PredictRequest req;
    req.set_request_id(42);
    // lang_tokens: [1, 200] 个非负整数
    for (int i = 1; i <= 4; ++i) req.add_lang_tokens(i * 3);
    // state: 与 max_state_dim(32) 匹配的浮点数
    for (int i = 0; i < 32; ++i) req.add_state(0.1f * i);
    // images: 一张 224x224 RGB_U8 渐变图（触发视觉编码，走完整推理）
    {
        const int W = 224, H = 224;
        vla::Image *img = req.add_images();
        img->set_encoding(vla::Image::RGB_U8);
        img->set_height(H);
        img->set_width(W);
        std::string px;
        px.reserve(static_cast<size_t>(W) * H * 3);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                px.push_back(static_cast<char>((x * 255) / W));
                px.push_back(static_cast<char>((y * 255) / H));
                px.push_back(static_cast<char>(((x + y) * 255) / (W + H)));
            }
        img->set_data(px);
    }

    std::string data;
    if (!req.SerializeToString(&data)) {
        fprintf(stderr, "SerializeToString failed\n");
        return 1;
    }
    fprintf(stdout, "sending %zu bytes to %s ...\n", data.size(), addr);
    if (zmq_send(sock, data.data(), data.size(), 0) != static_cast<int>(data.size())) {
        fprintf(stderr, "zmq_send failed\n");
        return 1;
    }

    char buf[1 << 22];
    const int n = static_cast<int>(zmq_recv(sock, buf, sizeof(buf), 0));
    if (n < 0) {
        fprintf(stderr, "zmq_recv failed/timed out (server not responding?)\n");
        return 1;
    }

    vla::PredictResponse resp;
    if (!resp.ParseFromArray(buf, n)) {
        fprintf(stderr, "ParseFromArray failed (%d bytes)\n", n);
        return 1;
    }
    fprintf(stdout, "request_id=%llu chunk=%u action_dim=%u err=\"%s\"\n",
            (unsigned long long) resp.request_id(), resp.chunk_size(), resp.action_dim(),
            resp.error().c_str());
    fprintf(stdout, "latency_ms: total=%.1f inference=%.1f prefill=%.1f denoise=%.1f vision=%.1f\n",
            resp.latency_ms_total(), resp.latency_ms_inference(), resp.latency_ms_prefill(),
            resp.latency_ms_denoise(), resp.latency_ms_vision());
    int shown = 0;
    fprintf(stdout, "action[0..7] =");
    for (float v : resp.action_chunk()) {
        if (shown++ < 8) fprintf(stdout, " %.4f", v);
    }
    fprintf(stdout, "\naction_chunk_size=%d\n", resp.action_chunk_size());

    zmq_close(sock);
    zmq_ctx_destroy(ctx);
    return 0;
}
