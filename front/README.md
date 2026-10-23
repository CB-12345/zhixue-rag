# 智学前端

React + TypeScript + Vite，采用浅蓝色界面。主要页面为知识库、AI 问答、学习笔记和历史会话；图谱与每日回顾位于“更多功能”。

## 本地开发

使用 Node.js 22，在本目录执行：

```bash
npm ci
npm run dev
```

API 代理配置见 `vite.config.ts`，后端默认在 `http://localhost:8000`。

## 检查与构建

```bash
npm run lint
npm run build
```

Docker 构建执行生产构建，生成的静态文件由 Nginx 托管；lint 由开发命令和 CI 单独检查。整套应用的启动及模型配置见[项目 README](../README.md)。
