#ifndef VLA_SERVING_SERVER_BASE_H
#define VLA_SERVING_SERVER_BASE_H

#include <atomic>
#include <csignal>
#include <cstdio>
#include <string>
#include <zmq.hpp>

namespace vla {
namespace serving {

// ZeroMQ REP server base class that encapsulates the common
// poll / recv / send / shutdown loop shared by all RoboRT servers.
//
// Usage:
//   class MyServer : public ZmqServerBase {
//       std::string handle_request(const uint8_t * data, size_t size) override;
//   };
//   ...
//   MyServer srv("tcp://*:5555");
//   srv.run();  // blocks until SIGINT/SIGTERM
class ZmqServerBase {
  public:
    /// Construct and bind the ZMQ REP socket.
    explicit ZmqServerBase(const std::string & bind_addr);
    virtual ~ZmqServerBase();

    ZmqServerBase(const ZmqServerBase &)             = delete;
    ZmqServerBase & operator=(const ZmqServerBase &) = delete;

    /// Override to process one request.  Return the serialised response,
    /// or an empty string to skip sending a reply.
    virtual std::string handle_request(const uint8_t * data, size_t size) = 0;

    /// Blocking poll / recv / send loop.  Returns after signal_shutdown.
    void run();

    /// Number of successfully completed requests.
    uint64_t served() const { return served_; }

    /// Signal-safe shutdown trigger (install via std::signal).
    static void signal_shutdown(int /*signum*/) { s_shutdown_.store(true, std::memory_order_relaxed); }

    /// Programmatic shutdown.
    void shutdown() { s_shutdown_.store(true, std::memory_order_relaxed); }

  private:
    static std::atomic<bool> s_shutdown_;

  protected:
    zmq::context_t zctx_;
    zmq::socket_t  sock_;
    std::string    bind_addr_;
    uint64_t       served_ = 0;
};

}  // namespace serving
}  // namespace vla

#endif  // VLA_SERVING_SERVER_BASE_H
