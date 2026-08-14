import "./globals.css";

export const metadata = {
  title: "SigniaMAXX",
  description: "Real-time sign recognition",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
