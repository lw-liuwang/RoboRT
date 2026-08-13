#include "server_base.h"

namespace vla {
namespace serving {

std::atomic<bool> ZmqServerBase::s_shutdown_{ false };

ZmqServerBase::ZmqServerBase(const std::string & bind_addr, zmq::socket_type type) :
    zctx_(1),
    sock_(zctx_, type),
    bind_addr_(bind_addr) {
    int linger = 0;
    sock_.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
    sock_.bind(bind_addr_);
}

ZmqServerBase::~ZmqServerBase() = default;

void ZmqServerBase::run() {
    std::signal(SIGINT, signal_shutdown);
    std::signal(SIGTERM, signal_shutdown);

    zmq::pollitem_t poll[] = {
        { static_cast<void *>(sock_), 0, ZMQ_POLLIN, 0 }
    };

    std::printf("server_base: bound to %s. ready.\n", bind_addr_.c_str());

    while (!s_shutdown_.load(std::memory_order_relaxed)) {
        // ── poll ──────────────────────────────────────────────────────
        try {
            zmq::poll(poll, 1, std::chrono::milliseconds(200));
        } catch (const zmq::error_t & e) {
            if (e.num() == EINTR) {
                continue;
            }
            throw;
        }
        if (!(poll[0].revents & ZMQ_POLLIN)) {
            continue;
        }

        // ── recv ──────────────────────────────────────────────────────
        zmq::message_t req_msg;
        try {
            auto rr = sock_.recv(req_msg, zmq::recv_flags::none);
            if (!rr) {
                continue;
            }
        } catch (const zmq::error_t & e) {
            if (e.num() == EINTR) {
                continue;
            }
            throw;
        }

        // ── handle ────────────────────────────────────────────────────
        const std::string resp = handle_request(static_cast<const uint8_t *>(req_msg.data()), req_msg.size());

        // ── send (skip empty responses, e.g. when already sent in handler) ─
        if (!resp.empty()) {
            sock_.send(zmq::buffer(resp), zmq::send_flags::none);
        }
        ++served_;
    }

    std::printf("server_base: shutting down (served %llu requests)\n", (unsigned long long) served_);
}

}  // namespace serving
}  // namespace vla
